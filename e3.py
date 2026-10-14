"""E3 - ablations on V1, V2, V3. One curve per factor, the others held at defaults.
Factors: q, lambda, K, target N, single instant t vs mixed-across-blocks,
coherent tail (last stage + head) vs free, BN train vs eval.
"""

import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import chimera as CH
import stats as S
import runlib as R

EPOCH = 15
DRAWS = 3
KDEF = 16
BETA = C.BETA_DEFAULT
VICTIMS = ("V1", "V2", "V3")


def combined_registry(epochs):
    reg = {"resnet": [], "convnext": []}
    nets = {}
    for e in epochs:
        r, n = CH.load_donor_registry(e)
        for k in reg:
            reg[k] += r[k]
        nets.update(n)
    return reg, nets


def victim_stats(bn_mode):
    return {V: {s: R.stats_for_ckpt(V, s, EPOCH, bn_mode=bn_mode) for s in C.TRAIN_SEEDS}
            for V in VICTIMS}


def chim_eff(stV_by_vs, reg, nets, N, q, lam, bn_mode, coherent_tail, K, draws=DRAWS,
             reg_tag="t15"):
    out = {}
    for V in VICTIMS:
        effs = []
        for s in C.TRAIN_SEEDS:
            stV = stV_by_vs[V][s]
            for d in range(draws):
                base = d * 128 + 1
                sts = [R.chimera_stats(base + i, reg, nets, N, q=q, lam=lam, bn_mode=bn_mode,
                                       coherent_tail=coherent_tail, reg_tag=reg_tag)
                       for i in range(K)]
                ap = S.averaged_problem(sts)
                e, _ = S.eff(stV, S.solve_qp(ap["Q"], ap["c"], BETA), BETA)
                effs.append(e)
        out[V] = (float(np.mean(effs)), float(np.std(effs)))
    return out


def main():
    N = CH.default_N()
    reg, nets = CH.load_donor_registry(EPOCH)
    reg_mix, nets_mix = combined_registry(C.EVAL_EPOCHS)
    stV_train = victim_stats("train")
    stV_eval = victim_stats("eval")

    sweeps = {}  # factor -> (xvalues, {V: [(mean,std)...]})

    # q
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    sweeps["q"] = (xs, {V: [] for V in VICTIMS})
    for q in xs:
        r = chim_eff(stV_train, reg, nets, N, q, 0.3, "train", True, KDEF)
        for V in VICTIMS:
            sweeps["q"][1][V].append(r[V])

    # lambda
    xs = [0.0, 0.3, 0.6, 1.0]
    sweeps["lambda"] = (xs, {V: [] for V in VICTIMS})
    for lam in xs:
        r = chim_eff(stV_train, reg, nets, N, 0.5, lam, "train", True, KDEF)
        for V in VICTIMS:
            sweeps["lambda"][1][V].append(r[V])

    # K
    xs = C.K_VALUES
    sweeps["K"] = (xs, {V: [] for V in VICTIMS})
    for K in xs:
        r = chim_eff(stV_train, reg, nets, N, 0.5, 0.3, "train", True, K)
        for V in VICTIMS:
            sweeps["K"][1][V].append(r[V])

    # target N
    xs = [0.5, 1.0, 2.0]
    sweeps["N"] = (xs, {V: [] for V in VICTIMS})
    for f in xs:
        r = chim_eff(stV_train, reg, nets, int(f * N), 0.5, 0.3, "train", True, KDEF)
        for V in VICTIMS:
            sweeps["N"][1][V].append(r[V])

    # single vs mixed t   (categorical: 0=single, 1=mixed)
    sweeps["t_single_vs_mixed"] = ([0, 1], {V: [] for V in VICTIMS})
    r0 = chim_eff(stV_train, reg, nets, N, 0.5, 0.3, "train", True, KDEF)
    r1 = chim_eff(stV_train, reg_mix, nets_mix, N, 0.5, 0.3, "train", True, KDEF, reg_tag="tmix")
    for V in VICTIMS:
        sweeps["t_single_vs_mixed"][1][V] = [r0[V], r1[V]]

    # coherent tail vs free
    sweeps["coherent_tail"] = ([0, 1], {V: [] for V in VICTIMS})
    rf = chim_eff(stV_train, reg, nets, N, 0.5, 0.3, "train", False, KDEF)
    rc = chim_eff(stV_train, reg, nets, N, 0.5, 0.3, "train", True, KDEF)
    for V in VICTIMS:
        sweeps["coherent_tail"][1][V] = [rf[V], rc[V]]

    # BN train vs eval
    sweeps["bn_mode"] = ([0, 1], {V: [] for V in VICTIMS})
    rt = chim_eff(stV_train, reg, nets, N, 0.5, 0.3, "train", True, KDEF)
    re = chim_eff(stV_eval, reg, nets, N, 0.5, 0.3, "eval", True, KDEF)
    for V in VICTIMS:
        sweeps["bn_mode"][1][V] = [rt[V], re[V]]

    _figure(sweeps)
    torch.save(sweeps, os.path.join(C.STATS_DIR, "e3_results.pt"))
    for f, (xs, d) in sweeps.items():
        print("[E3] %s" % f)
        for V in VICTIMS:
            print("    %s: %s" % (V, ["%.2f" % m for (m, s) in d[V]]))
    return sweeps


def _figure(sweeps):
    n = len(sweeps)
    fig, ax = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    ax = ax.ravel()
    for k, (name, (xs, d)) in enumerate(sweeps.items()):
        a = ax[k]
        for V in VICTIMS:
            ys = [m for (m, s) in d[V]]; es = [s for (m, s) in d[V]]
            a.errorbar(range(len(xs)), ys, yerr=es, marker="o", capsize=3, label=V)
        a.set_xticks(range(len(xs))); a.set_xticklabels([str(x) for x in xs])
        a.axhline(1.0, color="k", lw=0.8); a.axhline(0.0, color="grey", lw=0.8, ls=":")
        a.set_title(name); a.set_ylabel("eff"); a.grid(alpha=0.3); a.legend(fontsize=7)
    for k in range(len(sweeps), len(ax)):
        ax[k].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "E3_ablations.png"), dpi=120)


if __name__ == "__main__":
    main()
