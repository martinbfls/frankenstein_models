"""Central configuration for the Frankenstein-surrogate study.

Question: can a gradient-based poisoning attack (Witches' Brew, FLIP) crafted on
"Frankenstein" surrogates -- networks reassembled from a bank of checkpoints of KNOWN
architectures and completed with random weights -- transfer to a victim whose
architecture is UNKNOWN (never in the bank)?
"""

import os
import torch

# ----------------------------------------------------------------------------- device
def pick_device():
    """cuda > mps (Apple silicon) > cpu. Override with FRANK_DEVICE=cpu|mps|cuda."""
    forced = os.environ.get("FRANK_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
NUM_WORKERS = 4 if DEVICE == "cuda" else 0     # spawn-based workers are slow / fragile on macOS

# ----------------------------------------------------------------------------- profile
# "full": the original cluster-sized study (unchanged).
# "mac" : same protocol with smaller networks / fewer epochs / smaller attack budgets, sized
#         for an Apple-silicon laptop (MPS). Chosen automatically when the device is MPS;
#         force with FRANK_PROFILE=full|mac. Checkpoints and results of the two profiles are
#         kept in separate folders, so they never mix.
def _auto_profile():
    forced = os.environ.get("FRANK_PROFILE")
    if forced:
        if forced not in ("full", "mac"):
            raise ValueError("FRANK_PROFILE must be 'full' or 'mac', got %r" % forced)
        return forced
    return "mac" if DEVICE == "mps" else "full"


PROFILE = _auto_profile()

# ----------------------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
DATA_DIR = os.environ.get("FRANK_DATA_DIR") or os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints" if PROFILE == "full" else "checkpoints_" + PROFILE)
RESULTS_DIR = os.path.join(ROOT, "results", "hypothesis" if PROFILE == "full" else "hypothesis_" + PROFILE)

# ----------------------------------------------------------------------------- dataset
# Same normalisation as Witches' Brew (forest/consts.py). FLIP uses another std; the FLIP
# loader rescales its inputs to this one (see flip/modules/base_utils/util.py).
NUM_CLASSES = 10
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# ----------------------------------------------------------------------------- training (bank + victims)
TRAIN_EPOCHS = 15 if PROFILE == "full" else 8
TRAIN_SEEDS = [0, 1, 2]        # 3 seeds per architecture
BATCH_SIZE = 128 if PROFILE == "full" else 256   # MPS is launch-latency bound: bigger batches = fewer steps
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

if PROFILE == "mac":   # same relations (A/V1 resnet, B/V2 convnext, V3 vgg), ~3-6x cheaper on MPS
    BANK_ARCHS = {
        "A": {"family": "resnet", "widths": (8, 16, 32), "blocks": (2, 2, 2)},
        "B": {"family": "convnext", "dims": (16, 32, 64), "depths": (1, 1, 1)},
    }
    VICTIM_ARCHS = {
        "V1": {"family": "resnet", "widths": (12, 24, 48), "blocks": (3, 3, 3)},
        "V2": {"family": "convnext", "dims": (20, 40, 80), "depths": (1, 2, 1)},
        "V3": {"family": "vgg", "cfg": (16, 16, "M", 32, 32, "M", 64, 64, "M")},
    }

# ----------------------------------------------------------------------------- Frankenstein generator
CHIMERA = {
    "donor_epoch": TRAIN_EPOCHS,  # all donors of one Frankenstein come from this single epoch
    "N_tol": 0.10,                # reject architectures with |P - N| / N > N_tol
    "q": 0.5,                     # per-block (and stem / head) inheritance probability
    "lam": 0.3,                   # scale of random weights feeding an inherited output channel
    "widths_set": (16, 24, 32, 48, 64, 96, 128) if PROFILE == "full" else (8, 12, 16, 24, 32, 48, 64),
    "n_stages": 3,
    "blocks_per_stage_set": (1, 2, 3, 4) if PROFILE == "full" else (1, 2, 3),
    "bn_recal_batches": 50,       # clean batches used to re-estimate BN running statistics
}

# Defaults of frankenstein.run_study (command-line flags override them)
RUN_DEFAULTS = dict(keys=4, K=4, restarts=4, attackiter=250, flip_budget=1500, flip_iterations=5)
if PROFILE == "mac":
    RUN_DEFAULTS = dict(keys=2, K=3, restarts=1, attackiter=60, flip_budget=1500, flip_iterations=2)

for _d in (DATA_DIR, CKPT_DIR):
    os.makedirs(_d, exist_ok=True)
