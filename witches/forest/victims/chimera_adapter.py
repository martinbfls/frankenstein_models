"""Bridge the chimera-toy project into the Witches' Brew framework.

Exposes three model builders usable from ``get_model``:
  - ``build_chimera(seed)``  : a fresh-init network whose architecture is sampled from the
    chimera grammar (sample_arch + assemble). Phase 1 trains it from scratch like any net;
    the donor weight-inheritance of study 1 is NOT used here.
  - ``build_named(name)``    : a victim (V1/V2/V3) or bank (A/B) architecture from config.

The chimera-toy package lives one level above the cloned repo; we add it to sys.path.
"""

import os
import sys
import numpy as np

_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import config as _C          # noqa: E402  (chimera-toy/config.py)
import models as _M          # noqa: E402  (chimera-toy/models.py)
import chimera as _CH        # noqa: E402  (chimera-toy/chimera.py)

_N_DEFAULT = None


def _default_N():
    global _N_DEFAULT
    if _N_DEFAULT is None:
        _N_DEFAULT = _CH.default_N()
    return _N_DEFAULT


def build_chimera(seed, num_classes=10):
    """Fresh-init network with an architecture drawn from the chimera grammar."""
    rng = np.random.default_rng(int(seed))
    spec, net, _ = _CH.sample_arch(rng, _default_N())
    net.chimera_spec = spec   # keep for logging; harmless extra attribute
    return net


def build_named(name, num_classes=10):
    """A named config architecture (victims V1/V2/V3 or bank A/B)."""
    arch = _C.VICTIM_ARCHS.get(name) or _C.BANK_ARCHS.get(name)
    if arch is None:
        raise ValueError("Unknown named architecture %r" % name)
    return _M.build_from_arch(arch, num_classes)


NAMED = ("V1", "V2", "V3", "A", "B")
