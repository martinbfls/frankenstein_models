"""Clean-label backdoor poisoning by gradient matching (Sleeper-Agent style).

Core pieces, surrogate-agnostic (a surrogate is any torch model on DEVICE):

  g_bd   = mean_x grad_theta  L( f(x_source + trigger), target )          (target direction)
  g_tr   = mean_i grad_theta  L( f(x_poison_i + delta_i), target )         (poison direction)
  craft  : delta* = argmin_delta  mean over surrogates [ 1 - cos(g_tr, g_bd) ],  ||delta||_inf <= eps

The poison images are TARGET-class images, their labels stay target (clean-label); only
delta is optimised. The same delta* is then baked into a fresh victim's training set and
the victim is trained from scratch; we read the attack success rate (ASR) and clean accuracy.

Everything is done on normalised tensors; the L_inf budget is expressed per channel as
eps/std so it equals `eps` in pixel space, exactly like data.trigger_delta.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import config as C
import data as D
import models as M


# ----------------------------------------------------------------------------- normalised tensor dataset
_MEAN = torch.tensor(C.CIFAR_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(C.CIFAR_STD).view(1, 3, 1, 1)


def eps_norm(eps=None):
    """Per-channel L_inf bound in normalised space for a pixel-space budget `eps`."""
    eps = C.POISON["eps"] if eps is None else eps
    return (eps / _STD).view(1, 3, 1, 1)


def valid_range():
    """Min/max of a valid normalised pixel (image in [0,1] before normalisation)."""
    lo = (0.0 - _MEAN) / _STD
    hi = (1.0 - _MEAN) / _STD
    return lo.view(1, 3, 1, 1), hi.view(1, 3, 1, 1)


def _materialise(train):
    """CIFAR-10 as one normalised float tensor [N,3,32,32] + long targets, no augmentation."""
    tag = os.path.join(C.DATA_DIR, "norm_%s.pt" % ("train" if train else "test"))
    if os.path.exists(tag):
        d = torch.load(tag, map_location="cpu")
        return d["x"], d["y"]
    ds = torchvision.datasets.CIFAR10(C.DATA_DIR, train=train, download=True)
    x = torch.from_numpy(ds.data).float().permute(0, 3, 1, 2) / 255.0   # [N,3,32,32] in [0,1]
    x = (x - _MEAN) / _STD
    y = torch.tensor(ds.targets, dtype=torch.long)
    torch.save({"x": x.contiguous(), "y": y}, tag)
    return x, y


def train_tensor():
    return _materialise(True)


def test_tensor():
    return _materialise(False)


# ----------------------------------------------------------------------------- trigger in normalised space
def trigger_vec(device):
    """The fixed trigger as a [1,3,32,32] normalised additive tensor."""
    return D.trigger_delta((1, 3, 32, 32), device)


# ----------------------------------------------------------------------------- gradients
def _params(model):
    return [p for p in model.parameters() if p.requires_grad]


def _set_bn(model, mode):
    model.train(mode == "train")


def flat_grad(model, loss, create_graph=False):
    g = torch.autograd.grad(loss, _params(model), create_graph=create_graph)
    return torch.cat([t.reshape(-1) for t in g])


def backdoor_grad(model, x_src, bn_mode="eval"):
    """g_bd = grad_theta mean L(f(x_src + trigger), target). Returned detached, unit-free."""
    _set_bn(model, bn_mode)
    x = x_src + trigger_vec(x_src.device)
    y = torch.full((x.shape[0],), C.TARGET_CLASS, dtype=torch.long, device=x.device)
    loss = F.cross_entropy(model(x), y)
    g = flat_grad(model, loss, create_graph=False).detach()
    return g


def poison_grad(model, x_poison, bn_mode="eval"):
    """g_tr = grad_theta mean L(f(x_poison), target), differentiable wrt x_poison."""
    _set_bn(model, bn_mode)
    y = torch.full((x_poison.shape[0],), C.TARGET_CLASS, dtype=torch.long, device=x_poison.device)
    loss = F.cross_entropy(model(x_poison), y)
    return flat_grad(model, loss, create_graph=True)


# ----------------------------------------------------------------------------- crafting
def craft_poison(surrogates, x_base, x_src, eps=None, steps=None, lr=None,
                 bn_mode=None, log=None):
    """Optimise a single delta so the poison gradient aligns with the backdoor gradient,
    averaged over the surrogate ensemble. Returns (delta, final_mean_cos).

    surrogates : list of models already on DEVICE (crafting point theta).
    x_base     : [n,3,32,32] normalised TARGET-class images (labels implicitly target).
    x_src      : [m,3,32,32] normalised SOURCE-class images (trigger added inside).
    """
    eps = C.POISON["eps"] if eps is None else eps
    steps = C.POISON["craft_steps"] if steps is None else steps
    lr = C.POISON["craft_lr"] if lr is None else lr
    bn_mode = C.POISON["craft_bn"] if bn_mode is None else bn_mode
    dev = x_base.device
    en = eps_norm(eps).to(dev)
    lo, hi = valid_range()
    lo, hi = lo.to(dev), hi.to(dev)

    g_bd = [backdoor_grad(m, x_src, bn_mode) for m in surrogates]

    delta = torch.zeros_like(x_base, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    last = 0.0
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        coss = []
        for m, gb in zip(surrogates, g_bd):
            gt = poison_grad(m, x_base + delta, bn_mode)
            coss.append(F.cosine_similarity(gt, gb, dim=0))
        cos = torch.stack(coss).mean()
        (1.0 - cos).backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(min=-en, max=en)
            delta.data = (x_base + delta).clamp(min=lo, max=hi) - x_base
        last = float(cos.detach())
        if log is not None and (t % 20 == 0 or t == steps - 1):
            log("    craft step %3d  mean cos = %.4f" % (t, last))
    return delta.detach(), last


def random_delta(x_base, eps=None, seed=0):
    """A budget-matched random (sign) perturbation: the 'random poison' null."""
    eps = C.POISON["eps"] if eps is None else eps
    en = eps_norm(eps).to(x_base.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randint(0, 2, x_base.shape, generator=g).float() * 2 - 1
    s = s.to(x_base.device) * en
    lo, hi = valid_range()
    lo, hi = lo.to(x_base.device), hi.to(x_base.device)
    return (x_base + s).clamp(min=lo, max=hi) - x_base


# ----------------------------------------------------------------------------- poison selection
def poison_indices(y_train, n_poison=None, seed=None):
    """Indices of TARGET-class training images to poison (clean-label)."""
    n_poison = C.POISON["n_poison"] if n_poison is None else n_poison
    seed = C.POISON["poison_seed"] if seed is None else seed
    idx = torch.where(y_train == C.TARGET_CLASS)[0].numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    return torch.from_numpy(idx[:n_poison].copy())


def source_pool(x_train, y_train, n=None, seed=None):
    """A batch of SOURCE-class training images for estimating g_bd."""
    n = C.POISON["bd_batch"] if n is None else n
    seed = C.POISON["poison_seed"] if seed is None else seed
    idx = torch.where(y_train == C.SOURCE_CLASS)[0].numpy()
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(idx)
    return x_train[torch.from_numpy(idx[:n].copy())]


# ----------------------------------------------------------------------------- victim training / eval
def _augment(x):
    """Tensor-space RandomCrop(pad 4) + horizontal flip on a normalised batch."""
    n = x.shape[0]
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    ij = torch.randint(0, 9, (n, 2))
    out = torch.empty((n, 3, 32, 32), device=x.device)
    for k in range(n):               # small batches, tiny nets: explicit crop is fine
        i, j = int(ij[k, 0]), int(ij[k, 1])
        out[k] = x[k, :, i:i + 32, j:j + 32]
    flip = torch.rand(n, device=x.device) < 0.5
    out[flip] = torch.flip(out[flip], dims=[3])
    return out


def train_victim(arch, x_train, y_train, poison_idx=None, delta=None, epochs=None,
                 seed=0, aug=None, lr=None, log=None):
    """Train a fresh victim from scratch; poison_idx rows get delta added (labels unchanged)."""
    epochs = C.POISON["victim_epochs"] if epochs is None else epochs
    aug = C.POISON["victim_aug"] if aug is None else aug
    lr = C.POISON["victim_lr"] if lr is None else lr
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    xt = x_train.clone()
    if poison_idx is not None and delta is not None:
        lo, hi = valid_range()
        xt[poison_idx] = (xt[poison_idx] + delta.cpu()).clamp(min=lo, max=hi)
    net = M.build_from_arch(arch).to(C.DEVICE)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=C.MOMENTUM,
                          weight_decay=C.WEIGHT_DECAY, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = xt.shape[0]
    bs = C.BATCH_SIZE
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for b in range(0, n - bs + 1, bs):
            sel = perm[b:b + bs]
            xb = xt[sel].to(C.DEVICE)
            yb = y_train[sel].to(C.DEVICE)
            if aug:
                xb = _augment(xb)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(xb), yb).backward()
            opt.step()
        sched.step()
        if log is not None:
            log("    victim epoch %2d/%d done" % (ep + 1, epochs))
    return net


@torch.no_grad()
def clean_accuracy(net, x_test, y_test):
    net.eval()
    correct = total = 0
    for b in range(0, x_test.shape[0], 512):
        xb = x_test[b:b + 512].to(C.DEVICE)
        yb = y_test[b:b + 512].to(C.DEVICE)
        correct += (net(xb).argmax(1) == yb).sum().item()
        total += yb.numel()
    return correct / total


@torch.no_grad()
def attack_success(net, x_test, y_test):
    """ASR = fraction of SOURCE-class test images that, with the trigger, are read as TARGET.
    Also returns the 'natural' rate = those already read as target WITHOUT the trigger."""
    net.eval()
    idx = torch.where(y_test == C.SOURCE_CLASS)[0]
    xs = x_test[idx].to(C.DEVICE)
    trig = trigger_vec(C.DEVICE)
    hit = nat = total = 0
    for b in range(0, xs.shape[0], 512):
        xb = xs[b:b + 512]
        pt = net(xb + trig).argmax(1)
        pn = net(xb).argmax(1)
        hit += (pt == C.TARGET_CLASS).sum().item()
        nat += (pn == C.TARGET_CLASS).sum().item()
        total += xb.shape[0]
    return hit / total, nat / total
