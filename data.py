"""CIFAR-10 data, the fixed trigger, and the batches used by the matching problem."""

import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

import config as C


def _base_transform(train):
    aug = []
    if train:
        aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    return T.Compose(aug + [T.ToTensor(),
                            T.Normalize(C.CIFAR_MEAN, C.CIFAR_STD)])


def get_datasets():
    train = torchvision.datasets.CIFAR10(C.DATA_DIR, train=True, download=True,
                                         transform=_base_transform(True))
    test = torchvision.datasets.CIFAR10(C.DATA_DIR, train=False, download=True,
                                        transform=_base_transform(False))
    return train, test


def get_loaders(batch_size=C.BATCH_SIZE):
    train, test = get_datasets()
    tl = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                     num_workers=4, drop_last=True)
    vl = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False, num_workers=4)
    return tl, vl


# ----------------------------------------------------------------------------- trigger
_MEAN = torch.tensor(C.CIFAR_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(C.CIFAR_STD).view(1, 3, 1, 1)


def trigger_delta(shape, device):
    """Fixed additive sinusoidal vertical-stripe trigger in *normalized* space.

    Built in pixel space with amplitude EPS (L_inf <= EPS), then divided by the
    per-channel std so it adds directly to normalized tensors. NOT optimized.
    """
    _, _, h, w = shape
    xs = torch.arange(w, dtype=torch.float32)
    stripe = torch.sin(2 * math.pi * C.TRIGGER_FREQ * xs / w + C.TRIGGER_PHASE)  # [w] in [-1,1]
    delta_px = C.TRIGGER_EPS * stripe.view(1, 1, 1, w).repeat(1, 3, h, 1)        # pixel space
    delta = delta_px / _STD                                                       # normalized space
    return delta.to(device)


def apply_trigger(x):
    d = trigger_delta(x.shape, x.device)
    return x + d


# ----------------------------------------------------------------------------- per-class pools
def class_pools(n_per_class=C.IMAGES_PER_CLASS_STATS, seed=0):
    """Return a dict y -> tensor[n,3,32,32] of normalized (no-aug) training images."""
    ds = torchvision.datasets.CIFAR10(C.DATA_DIR, train=True, download=True,
                                      transform=_base_transform(False))
    targets = np.array(ds.targets)
    rng = np.random.default_rng(seed)
    pools = {}
    for y in range(C.NUM_CLASSES):
        idx = np.where(targets == y)[0]
        rng.shuffle(idx)
        idx = idx[:n_per_class]
        xs = torch.stack([ds[i][0] for i in idx])
        pools[y] = xs
    return pools


def clean_batch(pools, n=256, seed=1):
    """A class-balanced clean batch (x, y) drawn from the pools."""
    rng = np.random.default_rng(seed)
    per = max(1, n // C.NUM_CLASSES)
    xs, ys = [], []
    for y in range(C.NUM_CLASSES):
        p = pools[y]
        sel = rng.choice(len(p), size=per, replace=len(p) < per)
        xs.append(p[sel])
        ys.append(torch.full((per,), y, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


def backdoor_batch(pools):
    """Source-class images with the trigger applied, labelled as the target class."""
    x = pools[C.SOURCE_CLASS].clone()
    x = apply_trigger(x)
    y = torch.full((x.shape[0],), C.TARGET_CLASS, dtype=torch.long)
    return x, y
