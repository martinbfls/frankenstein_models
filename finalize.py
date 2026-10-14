"""Load E1/E2/E3 result caches, write the RESULTS.md sections and the decision verdict."""

import os
import numpy as np
import torch

import config as C

R_MD = os.path.join(C.ROOT, "RESULTS.md")


def _load(name):
    p = os.path.join(C.STATS_DIR, name)
    return torch.load(p, map_location="cpu", weights_only=False) if os.path.exists(p) else None


def e1_section(e1):
    if not e1:
        return "## E1\n_not run._\n"
    s = ["## E1 — are the statistics comparable across architectures?",
         "Epoch %d. Similarity group means:" % e1["epoch"], ""]
    s.append("| regime | cos(Q~) | rankcorr(c~) |")
    s.append("|---|---|---|")
    for k in ("real-real same arch", "real-real other arch", "chimera-victim"):
        q = e1["sQ"][k]; c = e1["sC"][k]
        s.append("| %s | %.3f ± %.3f | %.3f ± %.3f |" % (k, q[0], q[1], c[0], c[1]))
    s.append("")
    s.append("Chimera vs real-other-arch closeness to each victim (cos Q~):")
    for v, (sc, so) in e1["vcQ"].items():
        s.append("- %s: chimera=%.3f, real-other=%.3f → chimera %s" %
                 (v, sc, so, "closer" if sc > so else "NOT closer"))
    s.append("\nFigure: `figures/E1_similarity.png`\n")
    return "\n".join(s)


def e2_section(e2):
    if not e2:
        return "## E2\n_not run._\n"
    agg, betas, Ks = e2["agg"], e2["betas"], e2["Ks"]
    beta = C.BETA_DEFAULT if C.BETA_DEFAULT in betas else betas[0]
    Kmax = max(Ks)
    s = ["## E2 — does the solution transfer? (central test)",
         "Epoch %d, β=%g, K=%d. eff (mean ± std), cos(w,w*), top-10 overlap:" %
         (e2["epoch"], beta, Kmax), ""]
    s.append("| victim | chimera | real-ens | naive | chim q=0 | rand-init | heuristic | uniform |")
    s.append("|---|---|---|---|---|---|---|---|")
    for V in ("V1", "V2", "V3"):
        def g(m, K):
            t = agg.get((V, beta, m, K))
            return "%+.2f" % t[0] if t else "—"
        s.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            V, g("chimera", Kmax), g("real_ens", Kmax), g("naive", 0),
            g("chimera_q0", Kmax), g("randinit", Kmax), g("heuristic", 0), g("uniform", 0)))
    s.append("\nFigure: `figures/E2_transfer.png`\n")
    return "\n".join(s), agg, beta, Kmax


def verdict(agg, beta, Kmax):
    def val(V, m, K):
        t = agg.get((V, beta, m, K))
        return t[0] if t else float("nan")
    lines = ["## Decision"]
    verdicts = {}
    for V in ("V1", "V2", "V3"):
        ch = val(V, "chimera", Kmax); na = val(V, "naive", 0)
        re = val(V, "real_ens", Kmax); he = val(V, "heuristic", 0)
        q0 = val(V, "chimera_q0", Kmax); ri = val(V, "randinit", Kmax)
        tol = 0.05
        if abs(he - 1.0) < 0.05:
            vd = "USELESS (heuristic ≈ oracle: target too easy)"
        elif ch - ri < tol or ch - q0 < tol:
            vd = "WEAK (chimeras ≈ nulls: inherited content adds nothing)"
        elif ch > na + tol and ch >= re - tol and ch > he + tol:
            vd = "SUPPORTED"
        else:
            vd = "NOT SUPPORTED"
        verdicts[V] = vd
        lines.append("- **%s**: %s  (chimera=%+.2f, real-ens=%+.2f, naive=%+.2f, "
                     "heuristic=%+.2f, q0=%+.2f, rand-init=%+.2f)" %
                     (V, vd, ch, re, na, he, q0, ri))
    v1 = val("V1", "chimera", Kmax); v2 = val("V2", "chimera", Kmax); v3 = val("V3", "chimera", Kmax)
    if v3 < min(v1, v2) - 0.2:
        lines.append("- **Coverage note**: V3 (out-of-family) is far below V1/V2 → "
                     "coverage depends on the grammar (hypothesis strained out-of-vocabulary).")
    return "\n".join(lines) + "\n", verdicts


def e3_section(e3):
    if not e3:
        return "## E3\n_not run._\n"
    s = ["## E3 — ablations", "eff at β=%g, K=16 unless swept. Figure: `figures/E3_ablations.png`" %
         C.BETA_DEFAULT, ""]
    for f, (xs, d) in e3.items():
        s.append("- **%s** (x=%s):" % (f, xs))
        for V in ("V1", "V2", "V3"):
            s.append("    - %s: %s" % (V, [round(m, 2) for (m, _s) in d[V]]))
    s.append("")
    return "\n".join(s)


def main():
    e1 = _load("e1_results.pt"); e2 = _load("e2_results.pt"); e3 = _load("e3_results.pt")
    head = open(R_MD).read().split("## E1")[0].rstrip() + "\n\n"
    out = [head, e1_section(e1), "\n"]
    if e2:
        sec, agg, beta, Kmax = e2_section(e2)
        out += [sec, "\n"]
        vtxt, _ = verdict(agg, beta, Kmax)
        out += [vtxt, "\n"]
    else:
        out += ["## E2\n_not run._\n\n"]
    out += [e3_section(e3)]
    with open(R_MD, "w") as f:
        f.write("\n".join(out))
    print("[finalize] RESULTS.md written")


if __name__ == "__main__":
    main()
