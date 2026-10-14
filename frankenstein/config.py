"""Central configuration for the Frankenstein-surrogate study.

Question: can a gradient-based poisoning attack (Witches' Brew, FLIP) crafted on
"Frankenstein" surrogates -- networks reassembled from a bank of checkpoints of KNOWN
architectures and completed with random weights -- transfer to a victim whose
architecture is UNKNOWN (never in the bank)?
"""

import os
import torch

# ----------------------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")

# ----------------------------------------------------------------------------- device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------------- dataset
# Same normalisation as Witches' Brew (forest/consts.py). FLIP uses another std; the FLIP
# loader rescales its inputs to this one (see flip/modules/base_utils/util.py).
NUM_CLASSES = 10
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# ----------------------------------------------------------------------------- training (bank + victims)
TRAIN_EPOCHS = 15
TRAIN_SEEDS = [0, 1, 2]        # 3 seeds per architecture
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# ----------------------------------------------------------------------------- bank architectures (donors)
# Family "resnet": CIFAR ResNet with BasicBlock, homogeneous width per stage.
# Family "convnext": home-made ConvNeXt-tiny (dwconv 7x7, LN, expansion 4, layer scale).
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

# ----------------------------------------------------------------------------- Frankenstein generator
CHIMERA = {
    "donor_epoch": TRAIN_EPOCHS,  # all donors of one Frankenstein come from this single epoch
    "N_tol": 0.10,                # reject architectures with |P - N| / N > N_tol
    "q": 0.5,                     # per-block (and stem / head) inheritance probability
    "lam": 0.3,                   # scale of random weights feeding an inherited output channel
    "widths_set": (16, 24, 32, 48, 64, 96, 128),
    "n_stages": 3,
    "blocks_per_stage_set": (1, 2, 3, 4),
    "bn_recal_batches": 50,       # clean batches used to re-estimate BN running statistics
}

for _d in (DATA_DIR, CKPT_DIR):
    os.makedirs(_d, exist_ok=True)
