"""
Optimizes logit labels given expert trajectories using trajectory matching.
"""

from pathlib import Path
import copy
import sys

import torch
import numpy as np

from modules.base_utils.datasets import get_matching_datasets, pick_poisoner, get_n_classes
from modules.base_utils.util import extract_toml, get_module_device, get_mtt_attack_info,\
                                    load_model, either_dataloader_dataset_to_both, make_pbar,\
                                    needs_big_ims, slurmify_path, clf_loss, softmax,\
                                    total_mse_distance
from modules.generate_labels.utils import coalesce_attack_config, extract_experts,\
                                          extract_labels, sgd_step, DEFAULT_EXPERT_CONFIG


def run(experiment_name, module_name, **kwargs):
    """
    Optimizes and saves poisoned logit labels.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    :param kwargs: Additional arguments (such as slurm id).
    """

    slurm_id = kwargs.get('slurm_id', None)

    args = extract_toml(experiment_name, module_name)

    input_pths = args.get("input_pths", None)
    opt_pths = args.get("opt_pths", None)
    expert_model_flag = args["expert_model"]
    dataset_flag = args["dataset"]
    poisoner_flag = args["poisoner"]
    clean_label = args["source_label"]
    target_label = args["target_label"]
    lam = args.get("lambda", 0.0)
    train_pct = args.get("train_pct", 1.0)
    batch_size = args.get("batch_size", None)
    epochs = args.get("epochs", None)
    expert_config = args.get('expert_config', {})
    config = coalesce_attack_config(args.get("attack_config", {}))

    output_dir = slurmify_path(args["output_dir"], slurm_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build datasets and initialize labels
    print("Building datasets...")
    n_classes = get_n_classes(dataset_flag)
    poisoner = pick_poisoner(poisoner_flag,
                             dataset_flag,
                             target_label)

    big_ims = needs_big_ims(expert_model_flag)
    _, _, _, _, mtt_dataset =\
        get_matching_datasets(dataset_flag, poisoner, clean_label, train_pct=train_pct, big=big_ims)
    
    labels = extract_labels(mtt_dataset.distill, config['one_hot_temp'], n_classes)
    labels_init = torch.stack(extract_labels(mtt_dataset.distill, 1, n_classes))
    labels_syn = torch.stack(labels).requires_grad_(True)

    # Expert starting points. Two sources:
    #  - trajectories (input_pths / opt_pths given): checkpoints saved by train_expert;
    #  - Frankenstein experts (no input_pths): expert k is the network rebuilt from the
    #    checkpoint bank by expert_model.format(k) (e.g. "frank-{}"), used as is, with a
    #    fresh optimizer state (no momentum). No expert training.
    frank_experts = input_pths is None
    if frank_experts:
        if not expert_model_flag.startswith("frank-") or '{}' not in expert_model_flag:
            raise ValueError("Without input_pths, expert_model must be 'frank-{}[-q<val>]'.")
        n_experts = {**DEFAULT_EXPERT_CONFIG, **expert_config}['experts']
        print(f"Using {n_experts} Frankenstein experts rebuilt from the checkpoint bank...")
        expert_ids = np.random.randint(n_experts, size=config['iterations']).tolist()
    else:
        print("Loading expert trajectories...")
        expert_starts, expert_opt_starts, expert_ids = extract_experts(
            expert_config,
            input_pths,
            config['iterations'],
            expert_opt_path=opt_pths
        )

    # Optimize labels
    print("Training...")

    # expert_model may contain '{}', formatted with the expert index, so that experts of
    # different architectures (e.g. Frankenstein ensembles "frank-{}") can be mixed.
    def expert_flag(expert_id):
        return expert_model_flag.format(expert_id) if '{}' in expert_model_flag else expert_model_flag

    models = {}

    def get_models(flag):
        """(student, expert, expert optimizer, initial state_dict, fresh optimizer state)
        for an architecture flag, built once. The initial state_dict is the starting point
        of a Frankenstein expert (load_model is deterministic for a given flag)."""
        if flag not in models:
            student = load_model(flag, n_classes)
            expert = load_model(flag, n_classes)
            opt = torch.optim.SGD(expert.parameters(), **config['expert_kwargs'])
            start = {k: v.detach().clone() for k, v in expert.state_dict().items()}
            models[flag] = (student, expert, opt, start, copy.deepcopy(opt.state_dict()))
        return models[flag]

    student_model, expert_model = get_models(expert_flag(expert_ids[0]))[:2]
    device = get_module_device(student_model)

    batch_size, epochs, _, optimizer_labels = get_mtt_attack_info(
        expert_model.parameters(),
        labels_syn,
        config['expert_kwargs'],
        config['labels_kwargs'],
        batch_size=batch_size,
        epochs=epochs
    )

    mtt_dataloader, _ = either_dataloader_dataset_to_both(mtt_dataset,
                                                          batch_size=batch_size)

    losses = []
    with make_pbar(total=config['iterations'] * len(mtt_dataset)) as pbar:
        for i in range(config['iterations']):
            for x_t, y_t, x_d, y_true, idx in mtt_dataloader:
                # Prepare data
                y_d = labels_syn[idx]
                x_t, y_t, x_d, y_d = x_t.to(device), y_t.to(device), x_d.to(device), y_d.to(device)

                # Load parameters
                student_model, expert_model, optimizer_expert, start, fresh_opt =\
                    get_models(expert_flag(expert_ids[i]))
                if frank_experts:
                    checkpoint, state_dict = start, copy.deepcopy(fresh_opt)
                else:
                    checkpoint = torch.load(expert_starts[i])
                    state_dict = torch.load(expert_opt_starts[i])
                expert_model.load_state_dict(checkpoint)
                student_model.load_state_dict({k: v.clone() for k, v in checkpoint.items()})
                expert_start = [v.clone() for v in expert_model.parameters()]

                optimizer_expert.load_state_dict(copy.deepcopy(state_dict))
                # per-parameter optimizer state, empty when it has none yet (fresh optimizer)
                param_states = [state_dict['state'].get(j, {}) for j in range(len(expert_start))]

                # Take a single expert / poison step
                expert_model.train()
                expert_model.zero_grad()
                loss = clf_loss(expert_model(x_t), y_t)
                loss.backward()
                optimizer_expert.step()
                expert_model.eval()

                # Train a single student step
                student_model.train()
                student_model.zero_grad()

                loss = clf_loss(student_model(x_d), softmax(y_d))
                grads = torch.autograd.grad(loss, student_model.parameters(), create_graph=True)

                # Calculate loss
                param_loss = torch.tensor(0.0).to(device)
                param_dist = torch.tensor(0.0).to(device)

                for initial, student, expert, grad, state in zip(expert_start,
                                                                 student_model.parameters(),
                                                                 expert_model.parameters(),
                                                                 grads,
                                                                 param_states):
                    student_update = sgd_step(student, grad, state, state_dict['param_groups'][0])

                    param_loss += total_mse_distance(student_update, expert)
                    param_dist += total_mse_distance(initial, expert)

                # Add Regularization and calculate loss
                reg_term = lam * torch.linalg.vector_norm(softmax(labels_syn) - labels_init, ord=1, axis=1).mean()
                grand_loss = (param_loss / param_dist) + reg_term
                g_loss = grand_loss.item()

                # Optimize labels and learning rate
                optimizer_labels.zero_grad()
                grand_loss.backward()
                optimizer_labels.step()

                # Record training information
                losses.append(g_loss)
                pbar.update(batch_size)
                pbar_postfix = {
                    'g_loss': "%.4g" % np.mean(losses[-20:]),
                    'reg_term':"%.4g" % reg_term,
                }
                pbar.set_postfix(**pbar_postfix)

    # Save results
    print("Saving results...")
    y_true = torch.stack([mtt_dataset[i][3].detach() for i in range(len(mtt_dataset.distill))])
    np.save(output_dir + "labels.npy", labels_syn.detach().numpy())
    np.save(output_dir + "true.npy", y_true)
    np.save(output_dir + "losses.npy", losses)

if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
