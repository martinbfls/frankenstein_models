"""Models of the Frankenstein package (repository root) for FLIP.

Flags:
  - ``V1, V2, V3, A, B``        : named architectures of frankenstein/config.py, fresh init;
  - ``frank-<seed>``            : Frankenstein network reassembled from the checkpoint bank
                                  (deterministic given the seed and the bank);
  - ``frank-<seed>-q<val>``     : same with inheritance probability q=<val>.

The bank was trained with CIFAR_STD of frankenstein/config.py while FLIP normalises CIFAR
with another std (same mean): every model is wrapped in ``Rescale`` so that it sees inputs
in the bank's normalisation.
"""

import os
import sys
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)   # appended: the root's data/ etc. must not shadow anything

from frankenstein import config as C    # noqa: E402
from frankenstein import models as M    # noqa: E402
from frankenstein import chimera as CH  # noqa: E402

NAMED = ("V1", "V2", "V3", "A", "B")


def is_project_flag(flag):
    return flag in NAMED or flag.startswith("frank-")


class Rescale(torch.nn.Module):
    """x_bank = (x_flip * std_flip + mean - mean) / std_bank = x_flip * std_flip / std_bank."""

    def __init__(self, net, std_in):
        super().__init__()
        self.net = net
        self.register_buffer("scale", (torch.tensor(std_in) / torch.tensor(C.CIFAR_STD)).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net(x * self.scale)


def parse_frank_flag(flag):
    """'frank-<seed>[-q<val>]' -> (seed, q or None)."""
    parts = flag.split("-")
    seed, q = int(parts[1]), None
    for p in parts[2:]:
        if p.startswith("q"):
            q = float(p[1:])
        else:
            raise ValueError("Unknown Frankenstein flag %r" % flag)
    return seed, q


def load_project_model(flag, std_in):
    if flag in NAMED:
        arch = C.VICTIM_ARCHS.get(flag) or C.BANK_ARCHS.get(flag)
        net = M.build_from_arch(arch)
    else:
        seed, q = parse_frank_flag(flag)
        net, info = CH.build_frankenstein(seed, q=q)
        print(f"{flag}: spec {info['spec']} | params {info['params']} | rho {info['rho']:.3f}")
    return Rescale(net, std_in)
