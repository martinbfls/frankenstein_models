"""Shared helpers for the E1/E2/E3 experiment drivers and notebooks."""

import os
import glob
import numpy as np
import torch

import config as C
import models as M
import chimera as CH
import stats as S
import data as D


_POOLS = None


def pools():
    global _POOLS
    if _POOLS is None:
        _POOLS = D.class_pools(seed=0)
    return _POOLS


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    net = M.build_from_arch(ck["arch"])
    net.load_state_dict(ck["state_dict"])
    return net.to(C.DEVICE), ck


def ckpt_path(name, seed, epoch):
    return os.path.join(C.CKPT_DIR, "%s_s%d_e%d.pt" % (name, seed, epoch))


def stats_for_ckpt(name, seed, epoch, bn_mode="train"):
    tag = "stat_%s_s%d_e%d_%s" % (name, seed, epoch, bn_mode)

    def build():
        net, _ = load_model(ckpt_path(name, seed, epoch))
        return S.compute_stats(net, pools(), bn_mode=bn_mode)
    return S.get_stats_cached(tag, build)


_DEFN = None


def _default_N():
    global _DEFN
    if _DEFN is None:
        _DEFN = CH.default_N()
    return _DEFN


def chimera_stats(seed, reg, nets, N, q=None, lam=None, bn_mode="train",
                  single_t=True, coherent_tail=True, cache=True, reg_tag="t15"):
    # keep the default-case tag identical to earlier runs (preserves the E2 cache);
    # append discriminators only when N != default or a non-default registry is used.
    extra = ("" if N == _default_N() else "_N%d" % N) + ("" if reg_tag == "t15" else "_" + reg_tag)
    tag = "chim_s%d_q%s_l%s_%s_ct%d%s" % (seed, q, lam, bn_mode, int(coherent_tail), extra)

    def build():
        net, info = CH.make_chimera(seed, reg, nets, N, q=q, lam=lam,
                                    single_t=single_t, coherent_tail=coherent_tail)
        d = S.compute_stats(net, pools(), bn_mode=bn_mode)
        d["info"] = info
        return d
    if not cache:
        return build()
    return S.get_stats_cached(tag, build)


def available_epoch(name="A", seed=0):
    es = []
    for p in glob.glob(os.path.join(C.CKPT_DIR, "%s_s%d_e*.pt" % (name, seed))):
        es.append(int(p.rsplit("_e", 1)[1][:-3]))
    return max(es) if es else None


def rank_corr(a, b):
    """Spearman rank correlation via numpy (avoid extra deps)."""
    a = np.asarray(a); b = np.asarray(b)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d > 0 else 0.0


def cos_mat(Qa, Qb):
    a = (Qa.detach().cpu().numpy() if torch.is_tensor(Qa) else Qa).ravel()
    b = (Qb.detach().cpu().numpy() if torch.is_tensor(Qb) else Qb).ravel()
    return S.cos(a, b)


def qtilde(st):
    return st["Q"] / st["normv2"]


def ctilde(st):
    c = st["c"] / st["normv2"]
    return c.detach().cpu().numpy() if torch.is_tensor(c) else c
