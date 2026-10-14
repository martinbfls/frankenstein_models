"""Generator self-test (CPU is enough, needs the trained bank at CHIMERA['donor_epoch']).

(a) 200 Frankensteins: forward + backward, count_params == info['params'],
    parameter count within N_tol of the target N.
(b) identity reconstruction: bank net A_s0 rebuilt with every unit inherited from itself
    through identity injections must equal the checkpoint exactly (stem, blocks, head,
    BN buffers).
(c) BN recalibration runs and leaves the weights untouched.

Run from the repository root: python -m frankenstein.check_generator
"""

import os
import numpy as np
import torch

from . import config as C
from . import models as M
from . import chimera as CH


def main(n=200):
    epoch = C.CHIMERA["donor_epoch"]
    reg, nets = CH.donor_registry(epoch)
    N = CH.default_N()
    print("[check] donor epoch %d | target N = %d | donor blocks %s | donor nets %d"
          % (epoch, N, {k: len(v) for k, v in reg.items()}, len(nets)))

    # (a)
    x = torch.randn(8, 3, 32, 32)
    rhos, Ns, within = [], [], 0
    for i in range(n):
        net, info = CH.make_chimera(1000 + i, reg, nets, N)
        pc = M.count_params(net)
        assert pc == info["params"], (pc, info["params"])
        net(x).sum().backward()
        within += int(abs(pc - N) <= C.CHIMERA["N_tol"] * N)
        rhos.append(info["rho"]); Ns.append(pc)
    a_pass = within == n
    print("[check a] fwd+bwd OK | within N_tol %d/%d | rho mean %.3f | N mean %d -> %s"
          % (within, n, np.mean(rhos), int(np.mean(Ns)), "PASS" if a_pass else "FAIL"))

    # (b)
    archA = C.BANK_ARCHS["A"]
    spec_A = [{"family": "resnet", "width": w, "nblocks": nb}
              for w, nb in zip(archA["widths"], archA["blocks"])]
    net_id = "A_s0_e%d.pt" % epoch
    ref_sd = torch.load(os.path.join(C.CKPT_DIR, net_id), map_location="cpu")["state_dict"]
    chim, _ = CH.make_chimera(0, reg, nets, identity_arch=spec_A, identity_net=net_id)
    csd = chim.state_dict()
    assert set(csd) == set(ref_sd), set(csd) ^ set(ref_sd)
    max_err = max((csd[k].float() - ref_sd[k].float()).abs().max().item()
                  for k in ref_sd if not k.endswith("num_batches_tracked"))
    b_pass = max_err == 0.0
    print("[check b] identity reconstruction max|delta| = %.2e -> %s"
          % (max_err, "PASS" if b_pass else "FAIL"))

    # (c)
    net, _ = CH.make_chimera(7, reg, nets, N)
    net.eval()
    before = {k: v.clone() for k, v in net.named_parameters()}
    CH.recalibrate_bn(net, [torch.randn(16, 3, 32, 32) for _ in range(4)])
    c_pass = all(torch.equal(before[k], v) for k, v in net.named_parameters()) and not net.training
    print("[check c] BN recalibration -> %s" % ("PASS" if c_pass else "FAIL"))

    ok = a_pass and b_pass and c_pass
    print("[check] %s" % ("ALL PASS" if ok else "FAILURE"))
    return ok


if __name__ == "__main__":
    main()
