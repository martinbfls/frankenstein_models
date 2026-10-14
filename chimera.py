"""Chimera generator (version 1, minimal).

Grammar:  stem -> stages (residual blocks of families A/B, homogeneous width per
stage, widths from a discrete set) -> stride-2 reductions between stages ->
GAP + linear head, sampled until the parameter count is within N_tol of a target N.

Weight filling:  for each block a donor of the same family (a checkpoint block at a
single instant t) is inherited with probability q.  Tensors are copied on the
intersection of shapes -- kernel axes centered, channel axes by a *uniform injection*
that is SHARED by every tensor touching the same channel space (conv output, BN
gamma/beta, the next conv input).  Everything not copied is He-scaled Rademacher
weights; Rademacher entries on the input axis that perturb an inherited output channel
are scaled by lambda.  One instant t per chimera (epoch ~ exponential); the last stage
and the head come from one donor net, inherited coherently with probability q per block.

Logs rho (inherited fraction) and rho_coh (coherently inherited fraction).
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn

import config as C
import models as M


# ----------------------------------------------------------------------------- roles
def block_roles(family):
    if family == "resnet":
        return {
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
    return {  # convnext
        "dwconv.weight": ("dim", "single"), "dwconv.bias": ("dim", None),
        "norm.weight": ("dim", None), "norm.bias": ("dim", None),
        "pwconv1.weight": ("hidden", "dim"), "pwconv1.bias": ("hidden", None),
        "pwconv2.weight": ("dim", "hidden"), "pwconv2.bias": ("dim", None),
        "gamma": ("dim", None),
        "proj.0.weight": ("dim", None), "proj.0.bias": ("dim", None),
        "proj.1.weight": ("outdim", "dim"), "proj.1.bias": ("outdim", None),
    }


def head_roles(family):
    r = {"fc.weight": ("class", "dim"), "fc.bias": ("class", None)}
    if family == "convnext":
        r.update({"norm.weight": ("dim", None), "norm.bias": ("dim", None)})
    return r


def space_sizes(family, in_ch, out_ch):
    if family == "resnet":
        return {"in": in_ch, "out": out_ch, "single": 1, "class": C.NUM_CLASSES}
    return {"dim": in_ch, "hidden": 4 * in_ch, "outdim": out_ch,
            "single": 1, "class": C.NUM_CLASSES}


# ----------------------------------------------------------------------------- injection + copy
def injection(T, Dn, rng, identity=False):
    """Index pairs (dst_idx, src_idx) mapping donor channels into target channels."""
    if identity:
        L = min(T, Dn)
        return np.arange(L), np.arange(L)
    if T >= Dn:
        return np.sort(rng.choice(T, Dn, replace=False)), np.arange(Dn)
    return np.arange(T), np.sort(rng.choice(Dn, T, replace=False))


def _center(dh, sh):
    k = min(dh, sh)
    return (dh - k) // 2, (sh - k) // 2, k


def copy_into(dst, inh, src, out_pair, in_pair):
    """Copy src into dst (both numpy) on channel injection + centered spatial.
    Mark written entries in the boolean mask `inh`. Returns (dst_out_idx, dst_in_idx)."""
    do, so = out_pair
    if dst.ndim == 1:
        dst[do] = src[so]
        inh[do] = True
        return do, None
    di, si = in_pair
    if dst.ndim == 2:
        dst[np.ix_(do, di)] = src[np.ix_(so, si)]
        inh[np.ix_(do, di)] = True
        return do, di
    # 4D conv: center spatial
    dh0, sh0, kh = _center(dst.shape[2], src.shape[2])
    dw0, sw0, kw = _center(dst.shape[3], src.shape[3])
    dr, sr = range(dh0, dh0 + kh), range(sh0, sh0 + kh)
    dc, sc = range(dw0, dw0 + kw), range(sw0, sw0 + kw)
    dst[np.ix_(do, di, dr, dc)] = src[np.ix_(so, si, sr, sc)]
    inh[np.ix_(do, di, dr, dc)] = True
    return do, di


def he_rademacher(shape, rng):
    fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
    alpha = np.sqrt(2.0 / max(1, fan_in))
    return (alpha * rng.choice([-1.0, 1.0], size=shape)).astype(np.float32)


def _neutral(name, shape):
    """Non-weight tensors get neutral values rather than Rademacher."""
    if "running_var" in name or name.endswith("norm.weight") or \
       (name.endswith(".weight") and name.split(".")[-2] in ("bn1", "bn2")) or \
       name.endswith("downsample.1.weight") or name.endswith("proj.0.weight"):
        return np.ones(shape, dtype=np.float32)
    if name == "gamma":
        return np.full(shape, 1e-6, dtype=np.float32)
    return np.zeros(shape, dtype=np.float32)


def is_weight(name):
    return name.endswith("weight") and ("conv" in name or name == "fc.weight" or
                                         "pwconv" in name or "proj.1" in name)


# ----------------------------------------------------------------------------- donor bank
def load_donor_registry(epoch):
    """family -> list of donor blocks; and net_id -> ordered blocks + head (for tails)."""
    reg = {"resnet": [], "convnext": []}
    nets = {}
    for path in sorted(glob.glob(os.path.join(C.CKPT_DIR, "*_e%d.pt" % epoch))):
        base = os.path.basename(path)
        name = base.split("_s")[0]
        if name not in C.BANK_ARCHS:          # bank only
            continue
        ck = torch.load(path, map_location="cpu")
        arch = ck["arch"]
        net = M.build_from_arch(arch)
        net.load_state_dict(ck["state_dict"])
        sd = ck["state_dict"]
        blocks = []
        for i, meta in enumerate(net.block_meta):
            pref = "blocks.%d." % i
            params = {k[len(pref):]: v.numpy().copy() for k, v in sd.items()
                      if k.startswith(pref)}
            b = {"family": meta["family"], "in_ch": meta["in_ch"],
                 "out_ch": meta["out_ch"], "params": params, "net_id": base}
            reg[meta["family"]].append(b)
            blocks.append(b)
        head = {k[len("head."):]: v.numpy().copy() for k, v in sd.items()
                if k.startswith("head.")}
        nets[base] = {"family": arch["family"], "blocks": blocks, "head": head}
    return reg, nets


# ----------------------------------------------------------------------------- grammar
def sample_arch(rng, N_target, n_tries=2000):
    cfg = C.CHIMERA
    for _ in range(n_tries):
        widths = sorted(rng.choice(cfg["widths_set"], cfg["n_stages"], replace=True))
        fams = rng.choice(["resnet", "convnext"], cfg["n_stages"])
        nbs = rng.choice(cfg["blocks_per_stage_set"], cfg["n_stages"])
        spec = [{"family": f, "width": int(w), "nblocks": int(n)}
                for f, w, n in zip(fams, widths, nbs)]
        net = assemble(spec)
        P = M.count_params(net)
        if abs(P - N_target) <= cfg["N_tol"] * N_target:
            return spec, net, P
    return spec, net, P  # last attempt (rejected caller-side if needed)


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


# ----------------------------------------------------------------------------- filling
def fill_block(target_meta, donor, rng, q_inherit, lam, identity=False):
    """Return (new_params dict, inh_masks dict, inherited_bool). donor may be None."""
    fam = target_meta["family"]
    roles = block_roles(fam)
    tsz = space_sizes(fam, target_meta["in_ch"], target_meta["out_ch"])
    # build a reference target block to know exact tensor shapes/names
    if fam == "resnet":
        ref = M.BasicBlock(target_meta["in_ch"], target_meta["out_ch"], target_meta["stride"])
    else:
        ref = M.ConvNeXtBlock(target_meta["in_ch"], out_ch=target_meta["out_ch"],
                              stride=target_meta["stride"])
    ref_sd = ref.state_dict()
    inherit = donor is not None and (identity or rng.random() < q_inherit)

    # per-space injection (shared across tensors of that space)
    inj = {}
    inherited_dst = {sp: np.array([], dtype=int) for sp in tsz}
    if inherit:
        dsz = space_sizes(fam, donor["in_ch"], donor["out_ch"])
        for sp in tsz:
            d_idx, s_idx = injection(tsz[sp], dsz[sp], rng, identity=identity)
            inj[sp] = (d_idx, s_idx)
            inherited_dst[sp] = d_idx

    new, masks = {}, {}
    for name, t in ref_sd.items():
        shape = tuple(t.shape)
        if name.endswith("num_batches_tracked"):
            new[name] = np.zeros(shape, dtype=np.int64)
            masks[name] = np.zeros(shape, dtype=bool)
            continue
        base = he_rademacher(shape, rng) if is_weight(name) else _neutral(name, shape)
        inh = np.zeros(shape, dtype=bool)
        if inherit and name in donor["params"] and name in roles:
            osp, isp = roles[name]
            out_pair = inj[osp]
            in_pair = inj[isp] if isp is not None else (None, None)
            copy_into(base, inh, donor["params"][name], out_pair, in_pair)
        new[name] = base.astype(np.float32)
        masks[name] = inh

    # lambda-scale Rademacher inputs that perturb inherited output channels
    if inherit and lam != 1.0:
        for name, t in ref_sd.items():
            if not is_weight(name) or t.ndim < 2 or name not in roles:
                continue
            osp, isp = roles[name]
            if isp is None or isp == "single":
                continue
            out_inh = inherited_dst[osp]
            in_inh = set(inherited_dst[isp].tolist())
            if len(out_inh) == 0:
                continue
            w = new[name]
            for oc in out_inh:
                for ic in range(w.shape[1]):
                    if ic not in in_inh:
                        w[oc, ic] = w[oc, ic] * lam
    return new, masks, inherit, inherited_dst


def fill_head(family, dim, donor_net, rng, q_inherit, identity=False):
    ref = M.Head(dim, C.NUM_CLASSES, norm=(family == "convnext"))
    ref_sd = ref.state_dict()
    roles = head_roles(family)
    inherit = donor_net is not None and (identity or rng.random() < q_inherit)
    new, masks = {}, {}
    if inherit:
        donor_dim = donor_net["head"]["fc.weight"].shape[1]
        dim_inj = injection(dim, donor_dim, rng, identity=identity)
        cls_inj = injection(C.NUM_CLASSES, C.NUM_CLASSES, rng, identity=True)
    for name, t in ref_sd.items():
        shape = tuple(t.shape)
        base = he_rademacher(shape, rng) if is_weight(name) else _neutral("fc." + name, shape)
        inh = np.zeros(shape, dtype=bool)
        if inherit and name in donor_net["head"] and name in roles:
            osp, isp = roles[name]
            out_pair = cls_inj if osp == "class" else dim_inj
            in_pair = dim_inj if isp == "dim" else (None, None)
            copy_into(base, inh, donor_net["head"][name], out_pair, in_pair)
        new[name] = base.astype(np.float32)
        masks[name] = inh
    return new, masks, inherit


# ----------------------------------------------------------------------------- top level
def make_chimera(seed, reg, nets, N_target, q=None, lam=None,
                 single_t=True, coherent_tail=True, identity_arch=None):
    """Build one chimera. identity_arch (spec) + q=1 + identity reproduces a net exactly
    (used by E0-b). Returns (net, info)."""
    cfg = C.CHIMERA
    q = cfg["q"] if q is None else q
    lam = cfg["lam"] if lam is None else lam
    rng = np.random.default_rng(seed)

    if identity_arch is not None:
        spec, net = identity_arch, assemble(identity_arch)
    else:
        spec, net, _ = sample_arch(rng, N_target)

    meta = net.block_meta
    n_blocks = len(meta)
    # which blocks are in the coherent tail (last stage)
    last_fam = spec[-1]["family"]
    tail_start = n_blocks - spec[-1]["nblocks"]
    tail_net = None
    if coherent_tail and nets:
        same = [k for k, v in nets.items() if v["family"] == last_fam] or list(nets)
        tail_net = nets[rng.choice(same)] if same else None

    sd = net.state_dict()
    inh_total = total = coh_inh = 0
    n_inherited = 0

    for i, m in enumerate(meta):
        identity = identity_arch is not None
        in_tail = i >= tail_start
        if in_tail and tail_net is not None:
            # coherent: take the matching block from the tail donor (by position from end)
            pos = len(tail_net["blocks"]) - (n_blocks - i)
            donor = tail_net["blocks"][pos] if 0 <= pos < len(tail_net["blocks"]) \
                and tail_net["blocks"][pos]["family"] == m["family"] else None
            if donor is None and reg[m["family"]]:
                donor = reg[m["family"]][rng.integers(len(reg[m["family"]]))]
        else:
            pool = reg[m["family"]]
            donor = pool[rng.integers(len(pool))] if pool else None
            if identity and nets:
                # E0-b: donor is the corresponding block of the reference net
                donor = list(nets.values())[0]["blocks"][i]

        new, masks, inherited, _ = fill_block(m, donor, rng, q, lam, identity=identity)
        if inherited:
            n_inherited += 1
        for name, arr in new.items():
            sd["blocks.%d.%s" % (i, name)] = torch.from_numpy(arr) \
                if arr.dtype != np.int64 else torch.from_numpy(arr)
            e = masks[name].sum()
            total += masks[name].size
            inh_total += e
            if in_tail and inherited:
                coh_inh += e

    # head
    dim = meta[-1]["out_ch"] if meta else spec[-1]["width"]
    hnet = tail_net if coherent_tail else None
    if identity_arch is not None and nets:
        hnet = list(nets.values())[0]
    hnew, hmask, hinh = fill_head(net.family if net.family != "chimera" else last_fam,
                                  dim, hnet, rng, 1.0 if identity_arch is not None else q,
                                  identity=identity_arch is not None)
    for name, arr in hnew.items():
        sd["head.%s" % name] = torch.from_numpy(arr)
        total += hmask[name].size
        inh_total += hmask[name].sum()
        if hinh:
            coh_inh += hmask[name].sum()

    # stem stays fresh init (count toward total, not inherited)
    for k, v in sd.items():
        if k.startswith("stem."):
            total += v.numel()

    net.load_state_dict(sd)
    info = {"spec": spec, "params": M.count_params(net),
            "rho": inh_total / max(1, total), "rho_coh": coh_inh / max(1, total),
            "n_blocks": n_blocks, "n_inherited": n_inherited}
    return net, info


def default_N():
    return int(round(np.mean([M.count_params(M.build_from_arch(a))
                              for a in C.BANK_ARCHS.values()])))
