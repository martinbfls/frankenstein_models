"""The matching problem: gradients -> G (90 cols) -> Q, c -> QP over w in R^90 -> eff.

For a model M (BN in train mode by default):
  g_c      = grad of clean loss
  grad_bd  = grad of E_{X~class9} loss(f(x+delta), 4)
  mu_p     = (1-lam_p) g_c + lam_p grad_bd ;   v = mu_p - g_c
  h_{y,c}  = E[ grad loss(x, c) | Y=y ]        (100 vectors, ~500 imgs/class)
  g_{y,c}  = h_{y,c} - h_{y,y}
  G        = [ pi_y * g_{y,c} ]_{c != y}       (90 columns)
  Q = G^T G ,   c = G^T v  (built on GPU, G discarded)
  B2(w) = (||v||^2 - 2 c^T w + w^T Q w) / ||v||^2
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
import osqp

import config as C
import data as D


PAIRS = [(y, c) for y in range(C.NUM_CLASSES) for c in range(C.NUM_CLASSES) if c != y]  # 90


def _flat_grad(model, loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False,
                            allow_unused=True)
    out = []
    for gi, p in zip(g, params):
        out.append((torch.zeros_like(p) if gi is None else gi).reshape(-1))
    return torch.cat(out)


def _mean_grad(model, x, label, params, chunk=250):
    """Mean-over-samples gradient of CE loss with all targets set to `label`."""
    n = x.shape[0]
    acc = None
    for s in range(0, n, chunk):
        xb = x[s:s + chunk].to(C.DEVICE)
        yb = torch.full((xb.shape[0],), label, dtype=torch.long, device=C.DEVICE)
        loss = F.cross_entropy(model(xb), yb, reduction="sum")
        g = _flat_grad(model, loss, params)
        acc = g if acc is None else acc + g
    return acc / n


def compute_stats(model, pools, bn_mode="train", clean_seed=1, chunk=250):
    """Return dict with Q (90x90), c (90,), normv2 (scalar), all as torch GPU tensors."""
    model = model.to(C.DEVICE)
    model.train() if bn_mode == "train" else model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    # ---- v = lam_p (grad_bd - g_c)
    xc, yc = D.clean_batch(pools, n=256, seed=clean_seed)
    xc = xc.to(C.DEVICE); yc = yc.to(C.DEVICE)
    g_c = _flat_grad(model, F.cross_entropy(model(xc), yc), params)
    xb, yb = D.backdoor_batch(pools)
    grad_bd = _mean_grad(model, xb, C.TARGET_CLASS, params, chunk)
    v = C.LAMBDA_P * (grad_bd - g_c)
    normv2 = (v @ v).item()

    # ---- h_{y,c} then G columns pi_y * (h_{y,c} - h_{y,y})
    pi = 1.0 / C.NUM_CLASSES
    P = sum(p.numel() for p in params)
    G = torch.empty((P, len(PAIRS)), device=C.DEVICE)
    h_diag = {}
    for y in range(C.NUM_CLASSES):
        h_diag[y] = _mean_grad(model, pools[y], y, params, chunk)
    for j, (y, c) in enumerate(PAIRS):
        h_yc = _mean_grad(model, pools[y], c, params, chunk)
        G[:, j] = pi * (h_yc - h_diag[y])

    Q = G.T @ G
    cvec = G.T @ v
    del G
    return {"Q": Q, "c": cvec, "normv2": normv2, "P": P}


# ----------------------------------------------------------------------------- QP
def solve_qp(Q, c, beta, ridge=1e-9):
    """argmin_w ||v - Gw||^2 s.t. w>=0, sum w <= beta.  Q,c may be torch or numpy."""
    Q = Q.detach().cpu().numpy() if torch.is_tensor(Q) else np.asarray(Q)
    c = c.detach().cpu().numpy() if torch.is_tensor(c) else np.asarray(c)
    n = Q.shape[0]
    Pm = 2.0 * (Q + ridge * np.eye(n))
    q = -2.0 * c
    A = sparse.vstack([sparse.eye(n), sparse.csc_matrix(np.ones((1, n)))]).tocsc()
    l = np.concatenate([np.zeros(n), [-np.inf]])
    u = np.concatenate([np.full(n, np.inf), [beta]])
    m = osqp.OSQP()
    m.setup(P=sparse.csc_matrix(Pm), q=q, A=A, l=l, u=u, **C.OSQP_SETTINGS)
    res = m.solve()
    w = np.asarray(res.x, dtype=np.float64)
    w = np.clip(w, 0, None)
    s = w.sum()
    if s > beta:
        w *= beta / s
    return w


def b2(Q, c, normv2, w):
    Q = Q.detach().cpu().numpy() if torch.is_tensor(Q) else np.asarray(Q)
    c = c.detach().cpu().numpy() if torch.is_tensor(c) else np.asarray(c)
    return (normv2 - 2.0 * c @ w + w @ Q @ w) / normv2


def eff(stats_V, w, beta):
    """Transfer efficiency of w on victim V vs V's own oracle."""
    w_star = solve_qp(stats_V["Q"], stats_V["c"], beta)
    b2w = b2(stats_V["Q"], stats_V["c"], stats_V["normv2"], w)
    b2o = b2(stats_V["Q"], stats_V["c"], stats_V["normv2"], w_star)
    denom = 1.0 - b2o
    e = (1.0 - b2w) / denom if abs(denom) > 1e-12 else float("nan")
    return e, w_star


# ----------------------------------------------------------------------------- averaging
def averaged_problem(stats_list):
    """Q_tilde = mean(Q_M/||v_M||^2), c_tilde = mean(c_M/||v_M||^2)."""
    Qs = sum(s["Q"] / s["normv2"] for s in stats_list) / len(stats_list)
    cs = sum(s["c"] / s["normv2"] for s in stats_list) / len(stats_list)
    return {"Q": Qs, "c": cs, "normv2": 1.0}


# ----------------------------------------------------------------------------- heuristics / nulls
def w_heuristic_source_target(beta):
    w = np.zeros(len(PAIRS))
    w[PAIRS.index((C.SOURCE_CLASS, C.TARGET_CLASS))] = beta
    return w


def w_uniform(beta):
    return np.full(len(PAIRS), beta / len(PAIRS))


def cos(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def topk_overlap(w, w_star, k=10):
    a = set(np.argsort(w)[::-1][:k].tolist())
    b = set(np.argsort(w_star)[::-1][:k].tolist())
    return len(a & b) / k


# ----------------------------------------------------------------------------- caching
def cache_path(tag):
    return os.path.join(C.STATS_DIR, tag + ".pt")


def get_stats_cached(tag, build_fn):
    p = cache_path(tag)
    if os.path.exists(p):
        d = torch.load(p, map_location=C.DEVICE)
        return d
    d = build_fn()
    torch.save({"Q": d["Q"].cpu(), "c": d["c"].cpu(), "normv2": d["normv2"],
                "P": d["P"]}, p)
    return {k: (v.to(C.DEVICE) if torch.is_tensor(v) else v) for k, v in d.items()}
