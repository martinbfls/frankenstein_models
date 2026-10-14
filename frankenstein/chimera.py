"""Frankenstein generator: networks reassembled from a bank of checkpoints.

1. Architecture (grammar): stem -> n_stages stages of residual blocks (family "resnet" or
   "convnext", one family and one width per stage, widths sorted ascending) -> stride-2
   reduction at the first block of every stage s > 0 -> GAP + linear head. Sampled until
   the parameter count is within N_tol of a target N (default: mean bank size).

2. Weights: every unit (stem, each block, head) is inherited from a donor with
   probability q, the donor being a bank checkpoint (one epoch for all donors):
     - block  : a block of the same family (any position, any bank net); the blocks of
                the last stage ("tail") and the head come from ONE net of the tail's
                family, matched by position from the end (coherent tail);
     - stem   : the stem of the net that donated block 0;
     - head   : the head of the tail net.
   Tensors are copied on the intersection of shapes: kernel axes centred, channel axes
   through an injection (sorted random subset). Injections are drawn once per
   (target channel space, donor net, donor channel space) and shared by every tensor
   that touches that space, so the residual stream, consecutive blocks from the same
   donor net and the head stay aligned.
   Everything not copied: He-scaled Rademacher weights, neutral norms (scale 1, shift
   0), zero biases, layer-scale 1e-6. Random weights that feed an inherited output
   channel from a non-inherited input channel are scaled by lam.

3. BN running statistics are meaningless after the merge: call recalibrate_bn() on
   clean data before using the network in eval mode.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import models as M


# ----------------------------------------------------------------------------- tensor roles
# name -> (output channel space, input channel space or None)
RESNET_ROLES = {
    "conv1.weight": ("out", "in"),
    "bn1.weight": ("out", None), "bn1.bias": ("out", None),
    "bn1.running_mean": ("out", None), "bn1.running_var": ("out", None),
    "conv2.weight": ("out", "out"),
    "bn2.weight": ("out", None), "bn2.bias": ("out", None),
    "bn2.running_mean": ("out", None), "bn2.running_var": ("out", None),
    "downsample.0.weight": ("out", "in"),
    "downsample.1.weight": ("out", None), "downsample.1.bias": ("out", None),
    "downsample.1.running_mean": ("out", None), "downsample.1.running_var": ("out", None),
}
CONVNEXT_ROLES = {
    "dwconv.weight": ("dim", "single"), "dwconv.bias": ("dim", None),
    "norm.weight": ("dim", None), "norm.bias": ("dim", None),
    "pwconv1.weight": ("hidden", "dim"), "pwconv1.bias": ("hidden", None),
    "pwconv2.weight": ("dim", "hidden"), "pwconv2.bias": ("dim", None),
    "gamma": ("dim", None),
    "proj.0.weight": ("dim", None), "proj.0.bias": ("dim", None),
    "proj.1.weight": ("outdim", "dim"), "proj.1.bias": ("outdim", None),
}
STEM_ROLES = {   # chimera stem = Conv(3->w, no bias) + BN + ReLU
    "0.weight": ("stream", "rgb"),
    "1.weight": ("stream", None), "1.bias": ("stream", None),
    "1.running_mean": ("stream", None), "1.running_var": ("stream", None),
}
HEAD_ROLES = {"fc.weight": ("class", "dim"), "fc.bias": ("class", None),
              "norm.weight": ("dim", None), "norm.bias": ("dim", None)}

# initialisation of non-inherited entries (explicit lists, no substring matching)
WEIGHT_NAMES = {"conv1.weight", "conv2.weight", "downsample.0.weight",
                "dwconv.weight", "pwconv1.weight", "pwconv2.weight", "proj.1.weight",
                "fc.weight"}
STEM_WEIGHT_NAMES = {"0.weight"}
ONES_NAMES = {"bn1.weight", "bn2.weight", "downsample.1.weight", "norm.weight",
              "proj.0.weight", "bn1.running_var", "bn2.running_var",
              "downsample.1.running_var"}
STEM_ONES_NAMES = {"1.weight", "1.running_var"}


def block_roles(family):
    return RESNET_ROLES if family == "resnet" else CONVNEXT_ROLES


def space_sizes(family, in_ch, out_ch):
    if family == "resnet":
        return {"in": in_ch, "out": out_ch}
    return {"dim": in_ch, "hidden": 4 * in_ch, "outdim": out_ch}


def stream_spaces(block_meta):
    """(in_space, out_space) of every block. Space 0 is the stem output; a block opens a
    new space iff it has a projection shortcut (stride != 1 or in_ch != out_ch), otherwise
    its input and output are the same residual stream."""
    out, cur = [], 0
    for m in block_meta:
        nxt = cur + 1 if (m["stride"] != 1 or m["in_ch"] != m["out_ch"]) else cur
        out.append((cur, nxt))
        cur = nxt
    return out


# ----------------------------------------------------------------------------- injection + copy
def injection(T, Dn, rng, identity=False):
    """Index pairs (dst_idx, src_idx) mapping donor channels into target channels."""
    if identity:
        L = min(T, Dn)
        return np.arange(L), np.arange(L)
    if T >= Dn:
        return np.sort(rng.choice(T, Dn, replace=False)), np.arange(Dn)
    return np.arange(T), np.sort(rng.choice(Dn, T, replace=False))


class Injections:
    """One injection per (target space, donor net, donor space), drawn on first use."""

    def __init__(self, rng, identity=False):
        self.rng, self.identity, self.cache = rng, identity, {}

    def get(self, key, T, Dn):
        if key not in self.cache:
            self.cache[key] = injection(T, Dn, self.rng, identity=self.identity)
        return self.cache[key]


FIXED_SPACES = {"single": (np.arange(1), np.arange(1)),
                "rgb": (np.arange(3), np.arange(3)),
                "class": (np.arange(C.NUM_CLASSES), np.arange(C.NUM_CLASSES))}


def _center(dh, sh):
    k = min(dh, sh)
    return (dh - k) // 2, (sh - k) // 2, k


def copy_into(dst, inh, src, out_pair, in_pair):
    """Copy src into dst (numpy) on channel injections + centred spatial window, and mark
    the written entries in the boolean mask `inh`."""
    do, so = out_pair
    if dst.ndim == 1:
        dst[do] = src[so]
        inh[do] = True
        return
    di, si = in_pair
    if dst.ndim == 2:
        dst[np.ix_(do, di)] = src[np.ix_(so, si)]
        inh[np.ix_(do, di)] = True
        return
    dh0, sh0, kh = _center(dst.shape[2], src.shape[2])
    dw0, sw0, kw = _center(dst.shape[3], src.shape[3])
    dr, sr = np.arange(dh0, dh0 + kh), np.arange(sh0, sh0 + kh)
    dc, sc = np.arange(dw0, dw0 + kw), np.arange(sw0, sw0 + kw)
    dst[np.ix_(do, di, dr, dc)] = src[np.ix_(so, si, sr, sc)]
    inh[np.ix_(do, di, dr, dc)] = True


def he_rademacher(shape, rng):
    fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
    alpha = np.sqrt(2.0 / max(1, fan_in))
    return (alpha * rng.choice([-1.0, 1.0], size=shape)).astype(np.float32)


def _init_tensor(name, shape, rng, weights, ones):
    if name in weights:
        return he_rademacher(shape, rng)
    if name in ones:
        return np.ones(shape, dtype=np.float32)
    if name == "gamma":
        return np.full(shape, 1e-6, dtype=np.float32)
    return np.zeros(shape, dtype=np.float32)          # biases, running means


def fill_unit(ref_sd, roles, donor_params, inj, rng, lam, weights=WEIGHT_NAMES, ones=ONES_NAMES):
    """Fill one unit (block, stem or head). inj: role space -> (dst_idx, src_idx), or
    None when the unit is not inherited. Returns (arrays, inherited masks)."""
    new, masks = {}, {}
    for name, t in ref_sd.items():
        shape = tuple(t.shape)
        if name.endswith("num_batches_tracked"):
            new[name] = np.zeros(shape, dtype=np.int64)
            masks[name] = np.zeros(shape, dtype=bool)
            continue
        base = _init_tensor(name, shape, rng, weights, ones)
        inh = np.zeros(shape, dtype=bool)
        if inj is not None and name in roles and name in donor_params:
            osp, isp = roles[name]
            copy_into(base, inh, donor_params[name], inj[osp],
                      inj[isp] if isp is not None else None)
        new[name], masks[name] = base, inh

    # random inputs feeding an inherited output channel are scaled by lam
    if inj is not None and lam != 1.0:
        for name, w in new.items():
            if name not in weights or name not in roles:
                continue
            osp, isp = roles[name]
            if isp is None or isp in FIXED_SPACES:
                continue
            rows = inj[osp][0]
            cols = np.setdiff1d(np.arange(w.shape[1]), inj[isp][0])
            if len(rows) and len(cols):
                w[np.ix_(rows, cols)] *= lam
    return new, masks


# ----------------------------------------------------------------------------- donor bank
def load_donor_registry(epoch):
    """Bank checkpoints (*_e<epoch>.pt of BANK_ARCHS) -> (reg, nets), see registry_from_entries."""
    entries = []
    for path in sorted(glob.glob(os.path.join(C.CKPT_DIR, "*_e%d.pt" % epoch))):
        base = os.path.basename(path)
        if base.split("_s")[0] not in C.BANK_ARCHS:          # bank only, never victims
            continue
        ck = torch.load(path, map_location="cpu")
        entries.append((base, ck["arch"], ck["state_dict"]))
    return registry_from_entries(entries)


def registry_from_entries(entries):
    """(net_id, arch, state_dict) triples -> reg: family -> donor blocks,
    nets: net_id -> {family, blocks, stem, head, final_space}."""
    reg = {"resnet": [], "convnext": []}
    nets = {}
    for net_id, arch, sd in entries:
        sd = {k: v.detach().cpu() for k, v in sd.items()}
        net = M.build_from_arch(arch)
        net.load_state_dict(sd)

        def sub(prefix):
            return {k[len(prefix):]: v.numpy().copy() for k, v in sd.items() if k.startswith(prefix)}

        spaces = stream_spaces(net.block_meta)
        blocks = []
        for i, meta in enumerate(net.block_meta):
            b = {"family": meta["family"], "in_ch": meta["in_ch"], "out_ch": meta["out_ch"],
                 "spaces": spaces[i], "params": sub("blocks.%d." % i), "net_id": net_id}
            reg[meta["family"]].append(b)
            blocks.append(b)
        stem = sub("stem.")
        if arch["family"] != "resnet":       # only the first conv matches the chimera stem
            stem = {k: v for k, v in stem.items() if k == "0.weight"}
        nets[net_id] = {"family": arch["family"], "blocks": blocks, "stem": stem,
                        "head": sub("head."), "final_space": spaces[-1][1] if spaces else 0}
    return reg, nets


_REGISTRY = {}


def donor_registry(epoch=None):
    """Cached registry of the bank at `epoch` (default CHIMERA['donor_epoch'])."""
    epoch = C.CHIMERA["donor_epoch"] if epoch is None else epoch
    if epoch not in _REGISTRY:
        reg, nets = load_donor_registry(epoch)
        if not nets:
            raise FileNotFoundError("no bank checkpoint *_e%d.pt in %s; train it with "
                                    "`python -m frankenstein.train_bank bank`" % (epoch, C.CKPT_DIR))
        _REGISTRY[epoch] = (reg, nets)
    return _REGISTRY[epoch]


# ----------------------------------------------------------------------------- grammar
def sample_arch(rng, N_target, n_tries=2000):
    cfg = C.CHIMERA
    for _ in range(n_tries):
        widths = sorted(rng.choice(cfg["widths_set"], cfg["n_stages"], replace=True))
        fams = rng.choice(["resnet", "convnext"], cfg["n_stages"])
        nbs = rng.choice(cfg["blocks_per_stage_set"], cfg["n_stages"])
        spec = [{"family": str(f), "width": int(w), "nblocks": int(n)}
                for f, w, n in zip(fams, widths, nbs)]
        net = assemble(spec)
        P = M.count_params(net)
        if abs(P - N_target) <= cfg["N_tol"] * N_target:
            return spec, net, P
    return spec, net, P  # last attempt (caller may reject it)


def assemble(spec):
    stem = nn.Sequential(nn.Conv2d(3, spec[0]["width"], 3, 1, 1, bias=False),
                         nn.BatchNorm2d(spec[0]["width"]), nn.ReLU(inplace=True))
    blk, meta = [], []
    in_ch = spec[0]["width"]
    for s, st in enumerate(spec):
        for j in range(st["nblocks"]):
            stride = 2 if (s > 0 and j == 0) else 1
            out_ch = st["width"]
            if st["family"] == "resnet":
                blk.append(M.BasicBlock(in_ch, out_ch, stride))
            else:
                blk.append(M.ConvNeXtBlock(in_ch, out_ch=out_ch, stride=stride))
            meta.append({"family": st["family"], "in_ch": in_ch,
                         "out_ch": out_ch, "stride": stride})
            in_ch = out_ch
    head = M.Head(in_ch, C.NUM_CLASSES, norm=(spec[-1]["family"] == "convnext"))
    return M.Net(stem, blk, meta, head, "chimera")


_DEFAULT_N = None


def default_N():
    global _DEFAULT_N
    if _DEFAULT_N is None:
        _DEFAULT_N = int(round(np.mean([M.count_params(M.build_from_arch(a))
                                        for a in C.BANK_ARCHS.values()])))
    return _DEFAULT_N


# ----------------------------------------------------------------------------- top level
def make_chimera(seed, reg, nets, N_target=None, q=None, lam=None, coherent_tail=True,
                 identity_arch=None, identity_net=None):
    """Build one Frankenstein network (BN statistics NOT recalibrated). Returns (net, info).

    identity_arch (a spec) + identity_net (a key of `nets`) rebuilds that bank net exactly:
    every unit inherited from it with identity injections (generator self-test)."""
    cfg = C.CHIMERA
    q = cfg["q"] if q is None else q
    lam = cfg["lam"] if lam is None else lam
    rng = np.random.default_rng(seed)
    identity = identity_arch is not None
    names = sorted(nets)

    if identity:
        spec, net = identity_arch, assemble(identity_arch)
        ref = nets[identity_net]
    else:
        spec, net, _ = sample_arch(rng, default_N() if N_target is None else N_target)
    meta = net.block_meta
    n_blocks = len(meta)
    spaces = stream_spaces(meta)
    final_space = spaces[-1][1] if spaces else 0
    inj = Injections(rng, identity)
    sd = net.state_dict()
    pnames = {k for k, _ in net.named_parameters()}
    count = {"total": sum(p.numel() for p in net.parameters()), "inh": 0, "coh": 0}

    def write(prefix, new, masks, coherent):
        for name, arr in new.items():
            sd[prefix + name] = torch.from_numpy(arr)
            if prefix + name in pnames:
                e = int(masks[name].sum())
                count["inh"] += e
                count["coh"] += e if coherent else 0

    def inherit_draw():
        return identity or rng.random() < q

    # tail donor net: one net of the last stage's family
    tail_id = None
    if identity:
        tail_id = identity_net
    elif coherent_tail:
        same = [k for k in names if nets[k]["family"] == spec[-1]["family"]]
        tail_id = same[rng.integers(len(same))] if same else None
    tail_start = n_blocks - spec[-1]["nblocks"]

    # ---- blocks
    n_inherited, donor_ids = 0, []
    for i, m in enumerate(meta):
        fam = m["family"]
        donor = None
        if identity:
            donor = ref["blocks"][i]
        else:
            if i >= tail_start and tail_id is not None:
                tb = nets[tail_id]["blocks"]
                pos = len(tb) - (n_blocks - i)
                if 0 <= pos and tb[pos]["family"] == fam:
                    donor = tb[pos]
            if donor is None and reg[fam]:
                donor = reg[fam][rng.integers(len(reg[fam]))]
        inherit = donor is not None and inherit_draw()

        roles = block_roles(fam)
        ref_blk = (M.BasicBlock(m["in_ch"], m["out_ch"], m["stride"]) if fam == "resnet" else
                   M.ConvNeXtBlock(m["in_ch"], out_ch=m["out_ch"], stride=m["stride"]))
        block_inj = None
        if inherit:
            nid = donor["net_id"]
            (t_in, t_out), (d_in, d_out) = spaces[i], donor["spaces"]
            tsz = space_sizes(fam, m["in_ch"], m["out_ch"])
            dsz = space_sizes(fam, donor["in_ch"], donor["out_ch"])
            if fam == "resnet":
                block_inj = {"in": inj.get((t_in, nid, d_in), tsz["in"], dsz["in"]),
                             "out": inj.get((t_out, nid, d_out), tsz["out"], dsz["out"])}
            else:
                block_inj = {"dim": inj.get((t_in, nid, d_in), tsz["dim"], dsz["dim"]),
                             "outdim": inj.get((t_out, nid, d_out), tsz["outdim"], dsz["outdim"]),
                             "hidden": inj.get(("hidden", i), tsz["hidden"], dsz["hidden"])}
            block_inj.update(FIXED_SPACES)
            n_inherited += 1
            donor_ids.append(nid)
        else:
            donor_ids.append(None)
        new, masks = fill_unit(ref_blk.state_dict(), roles,
                               donor["params"] if inherit else {}, block_inj, rng, lam)
        write("blocks.%d." % i, new, masks, coherent=(inherit and i >= tail_start
                                                      and donor["net_id"] == tail_id))

    # ---- stem: from the net that donated block 0
    stem_id = identity_net if identity else (donor_ids[0] if donor_ids and donor_ids[0]
                                             else (names[rng.integers(len(names))] if names else None))
    stem_inherit = stem_id is not None and inherit_draw()
    stem_inj = None
    if stem_inherit:
        w0 = spec[0]["width"]
        d_w0 = nets[stem_id]["stem"]["0.weight"].shape[0]
        stem_inj = {"stream": inj.get((0, stem_id, 0), w0, d_w0), **FIXED_SPACES}
    new, masks = fill_unit(net.stem.state_dict(), STEM_ROLES,
                           nets[stem_id]["stem"] if stem_inherit else {}, stem_inj, rng, lam,
                           weights=STEM_WEIGHT_NAMES, ones=STEM_ONES_NAMES)
    write("stem.", new, masks, coherent=False)

    # ---- head: from the tail net (or any net of the last family without coherent tail)
    head_id = tail_id
    if head_id is None and not identity:
        same = [k for k in names if nets[k]["family"] == spec[-1]["family"]] or names
        head_id = same[rng.integers(len(same))] if same else None
    head_inherit = head_id is not None and inherit_draw()
    head_inj = None
    if head_inherit:
        hd = nets[head_id]
        dim = meta[-1]["out_ch"] if meta else spec[-1]["width"]
        head_inj = {"dim": inj.get((final_space, head_id, hd["final_space"]),
                                   dim, hd["head"]["fc.weight"].shape[1]), **FIXED_SPACES}
    new, masks = fill_unit(net.head.state_dict(), HEAD_ROLES,
                           nets[head_id]["head"] if head_inherit else {}, head_inj, rng, lam)
    write("head.", new, masks, coherent=head_inherit and head_id == tail_id)

    net.load_state_dict(sd)
    info = {"spec": spec, "params": M.count_params(net),
            "rho": count["inh"] / max(1, count["total"]),
            "rho_coh": count["coh"] / max(1, count["total"]),
            "n_blocks": n_blocks, "n_inherited": n_inherited,
            "stem_inherited": stem_inherit, "head_inherited": head_inherit,
            "tail_net": tail_id, "donors": donor_ids}
    return net, info


# ----------------------------------------------------------------------------- make it usable
@torch.no_grad()
def recalibrate_bn(net, batches, n_batches=None):
    """Re-estimate every BatchNorm running mean/var as an exact average over `n_batches`
    clean batches (iterable of tensors or of (x, ...) tuples). Weights are untouched."""
    bns = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return net
    n_batches = C.CHIMERA["bn_recal_batches"] if n_batches is None else n_batches
    was_training = net.training
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                    # cumulative moving average
    net.train()
    dev = next(net.parameters()).device
    for k, x in enumerate(batches):
        if k >= n_batches:
            break
        if isinstance(x, (list, tuple)):
            x = x[0]
        net(x.to(dev))
    for m, mo in zip(bns, momenta):
        m.momentum = mo
    net.train(was_training)
    return net


def build_frankenstein(seed, q=None, lam=None, epoch=None):
    """Frankenstein network from the bank on disk (BN not yet recalibrated)."""
    reg, nets = donor_registry(epoch)
    return make_chimera(seed, reg, nets, q=q, lam=lam)
