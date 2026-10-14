"""E1 - are the matching statistics comparable across architectures (no optimization)?

For every real model (bank + victims) and for chimeras we form Q~ = Q/||v||^2 and c~.
Similarity between two models = cosine of flattened Q~, and rank-correlation of the c~
profile over the 90 pairs. We compare three regimes:
  real-real same arch  /  real-real other arch  /  chimera-victim
Question: is chimera-victim closer to a victim than a real other-architecture model is?
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import chimera as CH
import runlib as R


def build_set(epoch, n_chimera=8):
    items = []  # (label, group, stats)
    for name in ("A", "B"):
        for s in C.TRAIN_SEEDS:
            items.append(("%s.s%d" % (name, s), name, R.stats_for_ckpt(name, s, epoch)))
    for v in ("V1", "V2", "V3"):
        for s in C.TRAIN_SEEDS:
            items.append(("%s.s%d" % (v, s), v, R.stats_for_ckpt(v, s, epoch)))
    reg, nets = CH.load_donor_registry(epoch)
    N = CH.default_N()
    for i in range(n_chimera):
        st = R.chimera_stats(7000 + i, reg, nets, N)
        items.append(("chim%d" % i, "chim", st))
    return items


def similarity_matrices(items):
    n = len(items)
    cosQ = np.zeros((n, n)); rankc = np.zeros((n, n))
    ct = [R.ctilde(it[2]) for it in items]
    for i in range(n):
        for j in range(n):
            cosQ[i, j] = R.cos_mat(R.qtilde(items[i][2]), R.qtilde(items[j][2]))
            rankc[i, j] = R.rank_corr(ct[i], ct[j])
    return cosQ, rankc


FAM = {"A": "resnet", "B": "convnext", "V1": "resnet", "V2": "convnext", "V3": "vgg"}


def group_of(gi, gj):
    vic = {"V1", "V2", "V3"}
    if "chim" in (gi, gj) and (gi in vic or gj in vic):
        return "chimera-victim"
    if gi == "chim" or gj == "chim":
        return None
    same = FAM.get(gi) == FAM.get(gj) and gi == gj
    other = FAM.get(gi) != FAM.get(gj)
    # a victim vs a bank model of its OWN family but different config counts as "other arch"
    if same:
        return "real-real same arch"
    if other:
        return "real-real other arch"
    return "real-real other arch"  # same family, different config


def summarize(items, M):
    groups = {"real-real same arch": [], "real-real other arch": [], "chimera-victim": []}
    labels = [it[1] for it in items]
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            g = group_of(labels[i], labels[j])
            if g in groups:
                groups[g].append(M[i, j])
    return {k: (float(np.mean(v)) if v else float("nan"),
                float(np.std(v)) if v else float("nan"), len(v)) for k, v in groups.items()}


def victim_closeness(items, M):
    """For each victim, mean similarity to chimeras vs to real other-architecture models."""
    labels = [it[1] for it in items]
    out = {}
    for v in ("V1", "V2", "V3"):
        vidx = [i for i, l in enumerate(labels) if l == v]
        chim = [i for i, l in enumerate(labels) if l == "chim"]
        other = [i for i, l in enumerate(labels)
                 if l in ("A", "B") and FAM[l] != FAM[v]]
        sc = np.mean([M[a, b] for a in vidx for b in chim]) if vidx and chim else np.nan
        so = np.mean([M[a, b] for a in vidx for b in other]) if vidx and other else np.nan
        out[v] = (float(sc), float(so))
    return out


def main(epoch=None, n_chimera=8):
    epoch = epoch or R.available_epoch() or 15
    print("[E1] epoch =", epoch)
    items = build_set(epoch, n_chimera)
    cosQ, rankc = similarity_matrices(items)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    labels = [it[0] for it in items]
    for a, Mx, t in ((ax[0], cosQ, "cos(Q~)"), (ax[1], rankc, "rankcorr(c~)")):
        im = a.imshow(Mx, vmin=-1, vmax=1, cmap="RdBu_r")
        a.set_xticks(range(len(labels))); a.set_xticklabels(labels, rotation=90, fontsize=6)
        a.set_yticks(range(len(labels))); a.set_yticklabels(labels, fontsize=6)
        a.set_title(t); fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "E1_similarity.png"), dpi=120)

    sQ = summarize(items, cosQ); sC = summarize(items, rankc)
    vcQ = victim_closeness(items, cosQ); vcC = victim_closeness(items, rankc)
    print("\n[E1] group means (cos Q~):")
    for k, (m, s, n) in sQ.items():
        print("   %-22s %.3f ± %.3f  (n=%d)" % (k, m, s, n))
    print("[E1] group means (rankcorr c~):")
    for k, (m, s, n) in sC.items():
        print("   %-22s %.3f ± %.3f  (n=%d)" % (k, m, s, n))
    print("\n[E1] victim closeness  (chimera vs real-other-arch), cos Q~:")
    for v, (sc, so) in vcQ.items():
        print("   %s: chimera=%.3f  real-other=%.3f  -> chimera %s" %
              (v, sc, so, "closer" if sc > so else "not closer"))
    torch.save({"cosQ": cosQ, "rankc": rankc, "labels": labels, "sQ": sQ, "sC": sC,
                "vcQ": vcQ, "vcC": vcC, "epoch": epoch},
               os.path.join(C.STATS_DIR, "e1_results.pt"))
    return sQ, sC, vcQ, vcC


if __name__ == "__main__":
    main()
