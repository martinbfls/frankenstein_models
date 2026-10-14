"""E0 - generator validity (go/no-go).

(a) 200 chimeras: forward+backward on a batch; assert count_params == info params
    and |P - N| <= N_tol.
(b) identity reconstruction: an architecture rebuilt with q=1, identity injections and
    its own checkpoint as the sole donor must equal the checkpoint exactly.
(c) histograms of rho and rho_coh, distribution of N.

If no bank checkpoints exist yet, fresh-initialised bank nets are written at a probe
epoch so the generator can be validated before training finishes.
"""

import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import models as M
import chimera as CH

PROBE_EPOCH = 15
FALLBACK_EPOCH = 99   # fresh-init donors used only if nothing trained yet


def ensure_donors():
    if glob.glob(os.path.join(C.CKPT_DIR, "*_e%d.pt" % PROBE_EPOCH)):
        return PROBE_EPOCH
    print("[E0] no trained checkpoints; writing fresh-init donors at epoch", FALLBACK_EPOCH)
    for name, arch in C.BANK_ARCHS.items():
        for seed in C.TRAIN_SEEDS:
            torch.manual_seed(seed)
            net = M.build_from_arch(arch)
            torch.save({"state_dict": net.state_dict(), "arch": arch, "name": name,
                        "seed": seed, "epoch": FALLBACK_EPOCH},
                       os.path.join(C.CKPT_DIR, "%s_s%d_e%d.pt" % (name, seed, FALLBACK_EPOCH)))
    return FALLBACK_EPOCH


def main():
    epoch = ensure_donors()
    reg, nets = CH.load_donor_registry(epoch)
    N = CH.default_N()
    print("[E0] target N =", N, "| donor blocks:",
          {k: len(v) for k, v in reg.items()}, "| donor nets:", len(nets))

    # (a) 200 chimeras
    x = torch.randn(8, 3, 32, 32, device=C.DEVICE)
    rhos, rhocs, Ns, ok_count = [], [], [], 0
    for i in range(200):
        net, info = CH.make_chimera(1000 + i, reg, nets, N)
        net = net.to(C.DEVICE)
        pc = M.count_params(net)
        assert pc == info["params"], (pc, info["params"])
        y = net(x); y.sum().backward()
        within = abs(pc - N) <= C.CHIMERA["N_tol"] * N
        ok_count += int(within)
        rhos.append(info["rho"]); rhocs.append(info["rho_coh"]); Ns.append(pc)
    a_pass = ok_count == 200
    print("[E0a] 200 chimeras fwd+bwd OK; params==P(G) OK; within N_tol: %d/200 -> %s"
          % (ok_count, "PASS" if a_pass else "FAIL"))

    # (b) identity reconstruction
    name0 = C.BANK_ARCHS["A"]
    spec_A = [{"family": "resnet", "width": w, "nblocks": nb}
              for w, nb in zip(name0["widths"], name0["blocks"])]
    ckA = torch.load(sorted(glob.glob(os.path.join(C.CKPT_DIR, "A_s0_e%d.pt" % epoch)))[0],
                     map_location="cpu")
    regA = {"resnet": [], "convnext": []}
    netA = M.build_from_arch(name0); netA.load_state_dict(ckA["state_dict"])
    sdA = ckA["state_dict"]
    blocksA = []
    for i, meta in enumerate(netA.block_meta):
        pref = "blocks.%d." % i
        blocksA.append({"family": meta["family"], "in_ch": meta["in_ch"],
                        "out_ch": meta["out_ch"],
                        "params": {k[len(pref):]: v.numpy().copy() for k, v in sdA.items()
                                   if k.startswith(pref)}, "net_id": "A_s0"})
    netsA = {"A_s0": {"family": "resnet", "blocks": blocksA,
                      "head": {k[len("head."):]: v.numpy().copy() for k, v in sdA.items()
                               if k.startswith("head.")}}}
    chim, info = CH.make_chimera(0, regA, netsA, N, q=1.0,
                                 identity_arch=spec_A, coherent_tail=False)
    # stem is fresh in chimera; compare blocks + head only
    csd, rsd = chim.state_dict(), ckA["state_dict"]
    max_err = 0.0
    for k in rsd:
        # stem is fresh by design; num_batches_tracked is an integer BN counter, not a weight
        if k.startswith("stem.") or k.endswith("num_batches_tracked"):
            continue
        max_err = max(max_err, (csd[k].float() - rsd[k].float()).abs().max().item())
    b_pass = max_err < 1e-6
    print("[E0b] identity reconstruction max|Δ| (blocks+head) = %.2e -> %s"
          % (max_err, "PASS" if b_pass else "FAIL"))

    # (c) histograms
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    ax[0].hist(rhos, bins=20); ax[0].set_title("rho (inherited)")
    ax[1].hist(rhocs, bins=20); ax[1].set_title("rho_coh (coherent)")
    ax[2].hist(Ns, bins=20); ax[2].axvline(N, color="r", ls="--")
    ax[2].set_title("param count N (red=target)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "E0_histograms.png"), dpi=110)
    print("[E0c] rho mean=%.3f  rho_coh mean=%.3f  N mean=%d (target %d)"
          % (np.mean(rhos), np.mean(rhocs), int(np.mean(Ns)), N))

    go = a_pass and b_pass
    print("\n[E0] GO" if go else "\n[E0] NO-GO")
    return go


if __name__ == "__main__":
    main()
