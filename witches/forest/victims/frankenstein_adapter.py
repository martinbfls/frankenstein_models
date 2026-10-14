"""Bridge between Witches' Brew and the Frankenstein package (repository root).

Model names understood by ``get_model`` (CIFAR10):
  - ``Frankenstein``         : network reassembled from the bank of checkpoints
                               (frankenstein.chimera, q and lam from frankenstein/config.py);
  - ``Frankenstein-q<val>``  : same with inheritance probability q=<val>
                               (``Frankenstein-q0`` = random-weights null, same grammar);
  - ``V1, V2, V3, A, B``     : named architectures of frankenstein/config.py, fresh init;
  - ``<name>-s<seed>``       : the same, loaded from the trained checkpoint
                               checkpoints/<name>_s<seed>_e<donor_epoch>.pt (brew with
                               ``--skip_clean_training``). Such checkpoints are excluded from
                               ``victim_alignment`` (no cosine read on the surrogate itself).

Frankenstein weights come from the bank: brew with ``--skip_clean_training`` (or a short
``--max_epoch``) so that Witches does not retrain them; ``recalibrate`` re-estimates their
BN statistics on clean data before brewing.
"""

import os
import sys
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)   # appended: the root's data/ etc. must not shadow anything

from frankenstein import config as _C    # noqa: E402
from frankenstein import models as _M    # noqa: E402
from frankenstein import chimera as _CH  # noqa: E402

NAMED = ("V1", "V2", "V3", "A", "B")


def is_frankenstein(name):
    return name == "Frankenstein" or name.startswith("Frankenstein-q")


def build_frankenstein(name, seed, num_classes=10):
    """Frankenstein network for model name ``Frankenstein[-q<val>]`` (BN not recalibrated)."""
    if num_classes != _C.NUM_CLASSES:
        raise ValueError("The checkpoint bank is CIFAR-10 only.")
    q =float(name[len("Frankenstein-q"):]) if name.startswith("Frankenstein-q") else None
    net, info = _CH.build_frankenstein(int(seed), q=q)
    net.frankenstein_info = info
    print(f"{name} seed {seed}: spec {info['spec']} | params {info['params']} | "
          f"rho {info['rho']:.3f} | inherited blocks {info['n_inherited']}/{info['n_blocks']}")
    return net


def build_named(name, num_classes=10):
    """A named config architecture (victims V1/V2/V3 or bank A/B), fresh init."""
    arch = _C.VICTIM_ARCHS.get(name) or _C.BANK_ARCHS.get(name)
    if arch is None:
        raise ValueError("Unknown named architecture %r" % name)
    return _M.build_from_arch(arch, num_classes)


# checkpoint names (<name>_s<seed>_e<epoch>) loaded as surrogates in this process
SURROGATE_CKPTS = set()


def build_pretrained(name, num_classes=10):
    """``<name>-s<seed>``: named architecture loaded from its trained bank/victim checkpoint."""
    arch_name, seed = name.split("-s")
    net = build_named(arch_name, num_classes=num_classes)
    ckpt = "%s_s%s_e%d" % (arch_name, seed, _C.CHIMERA["donor_epoch"])
    net.load_state_dict(torch.load(os.path.join(_C.CKPT_DIR, ckpt + ".pt"), map_location="cpu")["state_dict"])
    SURROGATE_CKPTS.add(ckpt)
    print(f"{name}: loaded checkpoint {ckpt}")
    return net


def recalibrate(victim, kettle):
    """Re-estimate the BN statistics of every Frankenstein model of a Witches victim on
    clean (normalised, non-augmented) training batches."""
    models = victim.models if hasattr(victim, "models") else [victim.model]
    for model in models:
        net = model.module if isinstance(model, torch.nn.DataParallel) else model
        if hasattr(net, "frankenstein_info"):
            home = next(net.parameters()).device          # ensembles keep members on CPU
            _CH.recalibrate_bn(net.to(victim.setup["device"]), kettle.trainloader)
            net.to(home).eval()
            print("Frankenstein BN statistics recalibrated on clean training data.")


# ----------------------------------------------------------------------------- transfer metric
def _flat(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def _poison_grad(net, kettle, poison_delta, params, device, batch_size=128):
    """Mean gradient of the training loss over the poison set (with delta if given)."""
    n = len(kettle.poisonset)
    total = None
    for s in range(0, n, batch_size):
        items = [kettle.poisonset[i] for i in range(s, min(n, s + batch_size))]
        x = torch.stack([it[0] for it in items]).to(device)
        y = torch.tensor([it[1] for it in items], device=device)
        if poison_delta is not None:
            x = x + poison_delta[s:s + len(items)].to(device)
        loss = F.cross_entropy(net(x), y, reduction="sum")
        g = _flat(torch.autograd.grad(loss, params))
        total = g if total is None else total + g
    return total / n


def victim_alignment(kettle, poison_delta, names, epochs, device):
    """cos(poison gradient, adversarial target gradient) measured on clean checkpoints of
    the VICTIM architectures (frankenstein.train_bank, all seeds, given epochs), i.e. the
    quantity Witches maximises on its surrogates, read on the true victim (checkpoints used
    as surrogates are skipped). Also returns
    the same cosine without delta (clean poison images) as baseline.
    Returns {ckpt_name: (cos_poisoned, cos_clean)}."""
    targets = torch.stack([d[0] for d in kettle.targetset]).to(device)
    intended = torch.tensor(kettle.poison_setup["intended_class"], device=device)
    out = {}
    for name in names:
        arch = _C.VICTIM_ARCHS.get(name) or _C.BANK_ARCHS.get(name)
        if arch is None:
            continue
        for seed in _C.TRAIN_SEEDS:
            for ep in epochs:
                path = os.path.join(_C.CKPT_DIR, "%s_s%d_e%d.pt" % (name, seed, ep))
                if not os.path.exists(path) or "%s_s%d_e%d" % (name, seed, ep) in SURROGATE_CKPTS:
                    continue
                net = _M.build_from_arch(arch)
                net.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
                net = net.to(device).eval()
                params = [p for p in net.parameters() if p.requires_grad]
                tg = _flat(torch.autograd.grad(F.cross_entropy(net(targets), intended), params))
                gp = _poison_grad(net, kettle, poison_delta, params, device)
                g0 = _poison_grad(net, kettle, None, params, device)
                out["%s_s%d_e%d" % (name, seed, ep)] = (F.cosine_similarity(tg, gp, dim=0).item(),
                                                        F.cosine_similarity(tg, g0, dim=0).item())
    return out
