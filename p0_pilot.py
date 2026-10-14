"""P0 - white-box pilot gate.

Question: does clean-label gradient-matching poisoning induce the source->target backdoor
AT ALL in this CIFAR-10 toy, in the easiest possible (white-box) setting?  If it does not
even work white-box, there is nothing to transfer and the chimera study is pointless.

For each victim architecture:
  surrogate   = a clean-trained checkpoint of the SAME architecture (white-box).
  craft delta white-box, train a fresh victim from scratch on the poisoned set.
  compare ASR against two nulls: no poison (trigger-only leak) and random budget-matched delta.

Gate: white-box ASR clearly above both nulls, clean accuracy preserved.
"""

import os
import sys
import torch

import config as C
import runlib as R
import poison as P

VICTIMS = ["V1", "V2"]
SURR_SEED = 0            # clean checkpoint used as crafting surrogate
VICTIM_SEED = 1         # fresh victim trained from scratch (different seed)
EPOCH = C.TRAIN_EPOCHS


def log(msg):
    print(msg, flush=True)


def run_victim(name, x_train, y_train, x_test, y_test, poison_idx, delta, tag):
    net = P.train_victim(C.VICTIM_ARCHS[name], x_train, y_train,
                         poison_idx=poison_idx, delta=delta, seed=VICTIM_SEED, log=None)
    acc = P.clean_accuracy(net, x_test, y_test)
    asr, nat = P.attack_success(net, x_test, y_test)
    log("  [%s / %s] clean_acc=%.3f  ASR=%.3f  (natural target rate=%.3f)" %
        (name, tag, acc, asr, nat))
    return {"clean_acc": acc, "asr": asr, "nat": nat}


def main():
    victims = sys.argv[1:] or VICTIMS
    x_train, y_train = P.train_tensor()
    x_test, y_test = P.test_tensor()
    poison_idx = P.poison_indices(y_train)
    x_base = x_train[poison_idx].to(C.DEVICE)
    x_src = P.source_pool(x_train, y_train).to(C.DEVICE)
    log("[P0] source=%d target=%d | n_poison=%d (%.2f%% of train) | eps=%.4f | bd_batch=%d"
        % (C.SOURCE_CLASS, C.TARGET_CLASS, len(poison_idx),
           100.0 * len(poison_idx) / x_train.shape[0], C.POISON["eps"], x_src.shape[0]))

    results = {}
    for name in victims:
        log("\n[P0] victim %s" % name)
        # white-box surrogate = clean checkpoint of the victim's own architecture
        surr, _ = R.load_model(R.ckpt_path(name, SURR_SEED, EPOCH))
        delta, cos = P.craft_poison([surr], x_base, x_src, log=log)
        linf_px = float((delta.cpu() * P._STD).abs().max())
        log("  craft done: mean cos(g_tr, g_bd) = %.4f | ||delta||_inf(px)=%.4f (budget %.4f)"
            % (cos, linf_px, C.POISON["eps"]))
        del surr
        torch.cuda.empty_cache()

        results[name] = {
            "cos": cos,
            "whitebox": run_victim(name, x_train, y_train, x_test, y_test,
                                   poison_idx, delta, "white-box poison"),
            "clean": run_victim(name, x_train, y_train, x_test, y_test,
                                None, None, "no poison (trigger-only null)"),
            "random": run_victim(name, x_train, y_train, x_test, y_test,
                                 poison_idx, P.random_delta(x_base), "random-delta null"),
        }

    torch.save(results, os.path.join(C.STATS_DIR, "p0_results.pt"))
    log("\n[P0] summary (ASR | clean_acc):")
    for name, r in results.items():
        log("  %s: white-box %.3f|%.3f   null-trigger %.3f|%.3f   null-random %.3f|%.3f   (craft cos %.3f)"
            % (name, r["whitebox"]["asr"], r["whitebox"]["clean_acc"],
               r["clean"]["asr"], r["clean"]["clean_acc"],
               r["random"]["asr"], r["random"]["clean_acc"], r["cos"]))
    return results


if __name__ == "__main__":
    main()
