"""General interface script to launch poisoning jobs."""

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

    poison_delta = witch.brew(model, data)
    brew_time = time.time()

    # Transfer of the gradient alignment, read on clean checkpoints of the victim architectures
    alignment = victim_alignment(data, poison_delta, args.vnet or args.net, args.align_epochs, setup['device'])
    for ckpt, (cos_p, cos_0) in alignment.items():
        print(f'Victim alignment {ckpt}: cos(poison, target) = {cos_p:.4f} (clean poisons: {cos_0:.4f})')
    align_stats = dict(victim_cos=np.mean([v[0] for v in alignment.values()]) if alignment else '',
                       victim_cos_clean=np.mean([v[1] for v in alignment.values()]) if alignment else '')

    if not args.pretrained and not args.skip_clean_training and args.retrain_from_init:
        stats_rerun = model.retrain(data, poison_delta)
    else:
        stats_rerun = None  # we dont know the initial seed for a pretrained model so retraining makes no sense

    if args.vnet is not None:  # Validate the transfer model given by args.vnet
        train_net = args.net
        args.net = args.vnet
        if args.vruns > 0:
            model = forest.Victim(args, setup=setup)
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None
        args.net = train_net
    else:  # Validate the main model
        if args.vruns > 0:
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None
    test_time = time.time()


    timestamps = dict(train_time=str(datetime.timedelta(seconds=train_time - start_time)).replace(',', ''),
                      brew_time=str(datetime.timedelta(seconds=brew_time - train_time)).replace(',', ''),
                      test_time=str(datetime.timedelta(seconds=test_time - brew_time)).replace(',', ''),
                      **align_stats)
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
