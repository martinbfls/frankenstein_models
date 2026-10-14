"""Central configuration. All defaults live here; scripts import from this module.

The study tests one hypothesis: a gradient-matching solution computed on random
"chimera" surrogate models transfers to a victim of unknown architecture, because
the matching problem depends on architecture only through statistics living in the
common 90-class-pair space. Everything is a CIFAR-10 toy so the whole protocol runs
in a few hours on a single GPU.
"""

import os
import torch

# ----------------------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
STATS_DIR = os.path.join(ROOT, "stats_cache")
FIG_DIR = os.path.join(ROOT, "figures")

# ----------------------------------------------------------------------------- device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED_BASE = 0

# ----------------------------------------------------------------------------- dataset
NUM_CLASSES = 10
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMAGES_PER_CLASS_STATS = 500   # images per class used to estimate class-pair gradients

# ----------------------------------------------------------------------------- toy matching problem
SOURCE_CLASS = 9               # source
TARGET_CLASS = 4               # target
LAMBDA_P = 0.05                # poison mix: mu_p = (1-lam)*g_c + lam*grad_bd
BETAS = [0.03, 0.01, 0.1]      # polytope budget W_beta = {Gw : w>=0, sum w <= beta}
BETA_DEFAULT = 0.03

# fixed additive trigger: sinusoidal vertical stripe, L_inf <= EPS, NOT optimized
TRIGGER_EPS = 8.0 / 255.0      # in input (pixel) space, before normalization
TRIGGER_FREQ = 6.0             # stripes across the image width
TRIGGER_PHASE = 0.0

# ----------------------------------------------------------------------------- training (bank + victims)
TRAIN_EPOCHS = 15
TRAIN_SEEDS = [0, 1, 2]        # 3 seeds per architecture
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
# victims are evaluated at these epochs (checkpoints saved every epoch anyway)
EVAL_EPOCHS = [2, 5, 10, 15]

# ----------------------------------------------------------------------------- bank architectures
# Family A: CIFAR ResNet with BasicBlock, homogeneous width per stage.
# Family B: home-made ConvNeXt-tiny (dwconv 7x7, LN, expansion 4, layer scale).
BANK_ARCHS = {
    "A": {"family": "resnet", "widths": (16, 32, 64), "blocks": (3, 3, 3)},
    "B": {"family": "convnext", "dims": (32, 64, 128), "depths": (2, 2, 2)},
}

# ----------------------------------------------------------------------------- victims (NEVER in the bank)
VICTIM_ARCHS = {
    "V1": {"family": "resnet", "widths": (24, 48, 96), "blocks": (4, 4, 4)},   # same family as A, unknown config
    "V2": {"family": "convnext", "dims": (40, 80, 160), "depths": (1, 3, 2)},  # same family as B, unknown config
    "V3": {"family": "vgg", "cfg": (32, 32, "M", 64, 64, "M", 128, 128, "M")}, # out-of-vocabulary family
}

# ----------------------------------------------------------------------------- chimera generator defaults
CHIMERA = {
    "N_target": None,          # set per-run; default computed from bank mean param count
    "N_tol": 0.10,             # reject |P(G) - N| / N > N_tol
    "q": 0.5,                  # per-block inheritance probability
    "lam": 0.3,                # scale in [0,1] for Rademacher inputs perturbing inherited channels
    "widths_set": (16, 24, 32, 48, 64, 96, 128),  # discrete width vocabulary
    "n_stages": 3,
    "blocks_per_stage_set": (1, 2, 3, 4),
    "t_exp_scale": 5.0,        # exponential law mean for the single instant t (epoch index)
    "alpha_mode": "he",        # Rademacher amplitude alpha from He scaling
}

# ----------------------------------------------------------------------------- clean-label poisoning (new study)
# Sleeper-Agent-style clean-label backdoor poisoning via gradient matching, with a
# chimera ensemble as the architecture-agnostic surrogate. One poison set must work
# across several in-family victim architectures (universality).
POISON = {
    "eps": 16.0 / 255.0,       # L_inf poison budget in pixel space (per channel)
    "n_poison": 500,           # number of poisoned images, all from the TARGET class (clean-label)
    "bd_batch": 512,           # source-class images (with trigger) used to estimate g_bd
    "craft_steps": 120,        # gradient-matching optimisation steps on delta
    "craft_lr": 0.05,          # Adam lr on delta
    "craft_bn": "eval",        # BN mode of surrogates while crafting
    "victim_epochs": 15,       # fresh victim training length
    "victim_aug": False,       # P0 best-case: no augmentation (poison not diluted by crops/flips)
    "victim_lr": 0.1,
    "poison_seed": 0,          # selection of the poison images + base for crafting
}

# experiment knobs
K_VALUES = [1, 4, 16, 64]      # ensemble sizes
N_CHIMERA_DRAWS = 5            # repetitions over chimera samplings
OSQP_SETTINGS = {"verbose": False, "eps_abs": 1e-7, "eps_rel": 1e-7, "max_iter": 20000,
                 "polishing": True}

for _d in (DATA_DIR, CKPT_DIR, STATS_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)
