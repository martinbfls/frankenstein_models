"""General interface script to launch poisoning jobs."""

import os
import csv
import torch
import numpy as np

import datetime
import time

import forest
from forest.victims.frankenstein_adapter import is_frankenstein, recalibrate, victim_alignment
torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()
# 100% reproducibility?
if args.deterministic:
    forest.utils.set_deterministic()


def already_recorded(args, victim_name):
    """True if the result table of this run already holds a row for (victim, poisonkey)."""
    fname = os.path.join(args.table_path, f'table_{"_".join(args.net)}_{args.threatmodel}{args.name}.csv')
    if not os.path.exists(fname):
        return False
    with open(fname) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    return any(r.get('victim') == victim_name and r.get('poisonkey') == str(args.poisonkey) for r in rows)


if __name__ == "__main__":

    setup = forest.utils.system_startup(args)

    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
    witch = forest.Witch(args, setup=setup)

    start_time = time.time()
    if any(is_frankenstein(name) for name in args.net):
        recalibrate(model, data)
    if args.pretrained:
        print('Loading pretrained model...')
        stats_clean = None
    elif args.skip_clean_training:
        print('Skipping clean training, brewing on the initial weights...')
        stats_clean = None
    else:
        stats_clean = model.train(data, max_epoch=args.max_epoch)
    train_time = time.time()

    if args.delta_cache is not None and os.path.exists(args.delta_cache):
        cache = torch.load(args.delta_cache, map_location='cpu')
        poison_delta, witch.stat_optimal_loss = cache['poison_delta'], cache['brewed_loss']
        print(f'Brewed poison loaded from {args.delta_cache}, brewing skipped.')
    else:
        poison_delta = witch.brew(model, data)
        if args.delta_cache is not None:
            os.makedirs(os.path.dirname(os.path.abspath(args.delta_cache)), exist_ok=True)
            torch.save(dict(poison_delta=poison_delta.cpu(), brewed_loss=witch.stat_optimal_loss),
                       args.delta_cache + '.tmp')
            os.replace(args.delta_cache + '.tmp', args.delta_cache)   # atomic: never a half-written cache
    brew_time = time.time()

    # Transfer of the gradient alignment, read on clean checkpoints of the victim architectures
    victims = args.vnet or args.net
    alignment = victim_alignment(data, poison_delta, victims, args.align_epochs, setup['device'])
    for ckpt, (cos_p, cos_0) in alignment.items():
        print(f'Victim alignment {ckpt}: cos(poison, target) = {cos_p:.4f} (clean poisons: {cos_0:.4f})')

    def align_stats(name):
        vals = [v for k, v in alignment.items() if k.startswith(name + '_s')]
        return dict(victim_cos=np.mean([v[0] for v in vals]) if vals else '',
                    victim_cos_clean=np.mean([v[1] for v in vals]) if vals else '')

    if not args.pretrained and not args.skip_clean_training and args.retrain_from_init:
        stats_rerun = model.retrain(data, poison_delta)
    else:
        stats_rerun = None  # we dont know the initial seed for a pretrained model so retraining makes no sense
    train_net, train_ensemble = args.net, args.ensemble
    test_time = time.time()

    # One validation and one table row per victim. Each victim is a single model trained from
    # scratch on the poisoned data (vruns times), never an ensemble of --ensemble copies.
    for victim_name in victims:
        if args.resume and already_recorded(args, victim_name):
            print(f'Victim {victim_name} already in the result table (poisonkey {args.poisonkey}), skipped.')
            continue
        if args.vnet is not None:  # Validate the transfer model given by args.vnet
            args.net, args.ensemble = [victim_name], 1
            if args.vruns > 0:
                stats_results = forest.Victim(args, setup=setup).validate(data, poison_delta)
            else:
                stats_results = None
            args.net, args.ensemble = train_net, train_ensemble
        else:  # Validate the main model
            stats_results = model.validate(data, poison_delta) if args.vruns > 0 else None
        test_time = time.time()

        timestamps = dict(train_time=str(datetime.timedelta(seconds=train_time - start_time)).replace(',', ''),
                          brew_time=str(datetime.timedelta(seconds=brew_time - train_time)).replace(',', ''),
                          test_time=str(datetime.timedelta(seconds=test_time - brew_time)).replace(',', ''),
                          victim=victim_name, poisonkey=args.poisonkey, **align_stats(victim_name))
        # Save run to table
        results = (stats_clean, stats_rerun, stats_results)
        forest.utils.record_results(data, witch.stat_optimal_loss, results,
                                    args, model.defs, model.model_init_seed, extra_stats=timestamps)

    # Export
    if args.save is not None:
        data.export_poison(poison_delta, path=args.poison_path, mode=args.save)

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
    print(f'--------------------------- brew time: {str(datetime.timedelta(seconds=brew_time - train_time))}')
    print(f'--------------------------- test time: {str(datetime.timedelta(seconds=test_time - brew_time))}')
    print('-------------Job finished.-------------------------')
