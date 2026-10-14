"""E2 - does the solution transfer to an unknown-architecture victim? (central test)

For each victim V, budget beta and victim seed we compute eff(w) for w from:
  1 oracle V            (= 1 by construction, the bound)
  2 naive transfer      (one real bank model of another architecture, same epoch)
  3 ensemble of K real checkpoints (uniform over A/B and epochs)  -- the decisive baseline
  4 K chimeras          (q=0.5, lam=0.3)
  5 K chimeras q=0      (null: does bank content matter?)
  6 K random-init real archs A/B (null 2)
  7 heuristic           (w = beta on pair 9->4)
  8 uniform w
Also cos(w, w*_V) and top-10 pair overlap. Mean ± std over 5 chimera draws x 3 victim seeds.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import models as M
import chimera as CH
import stats as S
import runlib as R

OTHER = {"V1": "B", "V2": "A", "V3": "A"}   # a bank arch of a different family


def real_ensemble(K, rng, epochs):
    picks = [(rng.choice(["A", "B"]), int(rng.choice(C.TRAIN_SEEDS)), int(rng.choice(epochs)))
             for _ in range(K)]
    sts = [R.stats_for_ckpt(n, s, e) for (n, s, e) in picks]
    return S.averaged_problem(sts)


def chimera_ensemble(K, base, reg, nets, N, q, lam):
    sts = [R.chimera_stats(base + i, reg, nets, N, q=q, lam=lam) for i in range(K)]
    return S.averaged_problem(sts)


def randinit_ensemble(K, base):
    sts = []
    for i in range(K):
        arch_name = "A" if (i % 2 == 0) else "B"
        tag = "randinit_%s_%d" % (arch_name, base + i)

        def build(an=arch_name, seed=base + i):
            torch.manual_seed(seed)
            net = M.build_from_arch(C.BANK_ARCHS[an]).to(C.DEVICE)
            return S.compute_stats(net, R.pools())
        sts.append(S.get_stats_cached(tag, build))
    return S.averaged_problem(sts)


def eval_w(stats_V, w, beta):
    e, wstar = S.eff(stats_V, w, beta)
    return e, S.cos(w, wstar), S.topk_overlap(w, wstar, 10)


def main(epoch=15, betas=None, victim_seeds=None, draws=5, Ks=None):
    betas = betas or C.BETAS
    victim_seeds = victim_seeds or C.TRAIN_SEEDS
    Ks = Ks or C.K_VALUES
    reg, nets = CH.load_donor_registry(epoch)
    N = CH.default_N()
    res = {}   # (V, beta, method, K) -> list of (eff, cos, overlap)

    def push(key, tup):
        res.setdefault(key, []).append(tup)

    for V in ("V1", "V2", "V3"):
        for vs in victim_seeds:
            stV = R.stats_for_ckpt(V, vs, epoch)
            for beta in betas:
                # deterministic-per-victim methods
                push((V, beta, "oracle", 0), eval_w(stV, S.solve_qp(stV["Q"], stV["c"], beta), beta))
                st_naive = R.stats_for_ckpt(OTHER[V], 0, epoch)
                w_naive = S.solve_qp(st_naive["Q"], st_naive["c"], beta)
                push((V, beta, "naive", 0), eval_w(stV, w_naive, beta))
                push((V, beta, "heuristic", 0), eval_w(stV, S.w_heuristic_source_target(beta), beta))
                push((V, beta, "uniform", 0), eval_w(stV, S.w_uniform(beta), beta))
                for d in range(draws):
                    rng = np.random.default_rng(1000 * d + vs)
                    for K in Ks:
                        base = d * 64
                        ap = chimera_ensemble(K, base, reg, nets, N, 0.5, 0.3)
                        push((V, beta, "chimera", K), eval_w(stV, S.solve_qp(ap["Q"], ap["c"], beta), beta))
                        ap0 = chimera_ensemble(K, 500000 + base, reg, nets, N, 0.0, 0.3)
                        push((V, beta, "chimera_q0", K), eval_w(stV, S.solve_qp(ap0["Q"], ap0["c"], beta), beta))
                        apr = real_ensemble(K, np.random.default_rng(7 * d + K + vs), C.EVAL_EPOCHS)
                        push((V, beta, "real_ens", K), eval_w(stV, S.solve_qp(apr["Q"], apr["c"], beta), beta))
                        api = randinit_ensemble(K, 900000 + base)
                        push((V, beta, "randinit", K), eval_w(stV, S.solve_qp(api["Q"], api["c"], beta), beta))
        print("[E2] done", V)

    agg = {k: (float(np.mean([t[0] for t in v])), float(np.std([t[0] for t in v])),
               float(np.mean([t[1] for t in v])), float(np.mean([t[2] for t in v])))
           for k, v in res.items()}
    torch.save({"agg": agg, "epoch": epoch, "betas": betas, "Ks": Ks},
               os.path.join(C.STATS_DIR, "e2_results.pt"))
    _figure(agg, betas, Ks)
    _print_table(agg, betas, Ks)
    return agg


def _figure(agg, betas, Ks):
    beta = C.BETA_DEFAULT if C.BETA_DEFAULT in betas else betas[0]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    methods = [("chimera", "o-"), ("chimera_q0", "s--"), ("real_ens", "^-"),
               ("randinit", "x--")]
    flat = ["naive", "heuristic", "uniform"]
    for a, V in zip(ax, ("V1", "V2", "V3")):
        for m, sty in methods:
            ys = [agg.get((V, beta, m, K), (np.nan,))[0] for K in Ks]
            a.plot(Ks, ys, sty, label=m)
        for m in flat:
            y = agg.get((V, beta, m, 0), (np.nan,))[0]
            a.axhline(y, ls=":", alpha=0.7, label=m)
        a.axhline(1.0, color="k", lw=1); a.set_xscale("log", base=2)
        a.set_title("%s (beta=%g)" % (V, beta)); a.set_xlabel("K"); a.grid(alpha=0.3)
    ax[0].set_ylabel("eff"); ax[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "E2_transfer.png"), dpi=120)


def _print_table(agg, betas, Ks):
    beta = C.BETA_DEFAULT if C.BETA_DEFAULT in betas else betas[0]
    Kmax = max(Ks)
    print("\n[E2] eff (mean±std) at beta=%g, K=%d :" % (beta, Kmax))
    for V in ("V1", "V2", "V3"):
        print(" ", V)
        for m, K in [("oracle", 0), ("chimera", Kmax), ("real_ens", Kmax), ("naive", 0),
                     ("chimera_q0", Kmax), ("randinit", Kmax), ("heuristic", 0), ("uniform", 0)]:
            t = agg.get((V, beta, m, K))
            if t:
                print("     %-12s eff=%+.3f ± %.3f  cos=%+.3f  top10=%.2f"
                      % (m, t[0], t[1], t[2], t[3]))


if __name__ == "__main__":
    main()
