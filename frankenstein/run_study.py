"""Whole study in one command: does an attack forged on Frankenstein surrogates transfer to a
victim of unknown architecture?

    python -m frankenstein.run_study                 # everything (resumable), then figures
    python -m frankenstein.run_study --dry-run       # print the plan and the commands only
    python -m frankenstein.run_study --plots-only    # figures + summary from existing results
    python -m frankenstein.run_study --attacks witches --no-asr   # cheapest: cosine only

Resuming: if the command is interrupted (Ctrl-C, crash, lost session), re-run exactly the same
command. Every finished unit is skipped: generator test, each brew (the brewed poison is cached
in witches/deltas/, so an interruption during victim retraining does not re-brew), each victim
evaluation (one table row per victim), each FLIP label optimisation, flip selection and user
training. Only the unit running at the time of the interruption is redone. The parameters of
the first run are stored in run_args.json; re-running with different ones is refused (use
another --out, or --force to mix them on purpose).

Steps:
  0. bank A,B + victims V1-V3 checkpoints (frankenstein.train_bank) if missing;
     generator self-test (frankenstein.check_generator).
  1. Witches' Brew, one brew per (surrogate arm, poison key), evaluated on every victim:
       whitebox   : the victim's own architecture, trained checkpoint <V>_s0 (K=1);
       frank      : K Frankenstein surrogates (bank weights, no retraining);
       frank_q0   : null, same grammar, no inherited weight (q=0);
       bankA/bankB: the 3 trained checkpoints of a known bank architecture (wrong arch).
     Metrics per victim: victim_cos - victim_cos_clean (gradient alignment gained on clean
     victim checkpoints never used as surrogates) and ASR (victim retrained from scratch).
  2. FLIP, same arms (experts = fixed starting points, fresh optimizer, no expert training),
     labels flipped at --flip-budget, user model trained from scratch -> ASR, clean acc.
  3. Figures (results/hypothesis/figures/*.png) and summary.md.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

from . import config as C

ROOT = C.ROOT
WITCHES = os.path.join(ROOT, "witches")
FLIP = os.path.join(ROOT, "flip")
VICTIMS = list(C.VICTIM_ARCHS)                         # V1, V2, V3
ARMS = ["whitebox", "frank", "frank_q0", "bankA", "bankB"]
ARM_LABELS = {"whitebox": "White-box", "frank": "Frankenstein", "frank_q0": "Frank. q=0",
              "bankA": "Bank A", "bankB": "Bank B"}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "hypothesis"))
    p.add_argument("--attacks", default="witches,flip", help="comma list among witches,flip")
    p.add_argument("--keys", type=int, default=4, help="Witches poison keys (one target each), 1..keys")
    p.add_argument("--K", type=int, default=4, help="number of Frankenstein surrogates / experts")
    p.add_argument("--restarts", type=int, default=4, help="Witches restarts (original: 8)")
    p.add_argument("--attackiter", type=int, default=250)
    p.add_argument("--budget", type=float, default=0.01, help="Witches poison fraction")
    p.add_argument("--eps", type=float, default=16)
    p.add_argument("--epochs", type=int, default=C.TRAIN_EPOCHS, help="victim retraining epochs (both attacks)")
    p.add_argument("--no-asr", action="store_true", help="Witches: cosine only, no victim retraining")
    p.add_argument("--flip-budget", type=int, default=1500)
    p.add_argument("--flip-iterations", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plots-only", action="store_true")
    p.add_argument("--force", action="store_true", help="resume even if the parameters differ from run_args.json")
    return p.parse_args()


# ----------------------------------------------------------------------------- execution helpers
def run(cmd, cwd, log, dry):
    """Run cmd, stdout+stderr to log. Returns True on success."""
    print("  $ (cd %s && %s)" % (os.path.relpath(cwd, ROOT) or ".", " ".join(cmd)))
    if dry:
        return True
    os.makedirs(os.path.dirname(log), exist_ok=True)
    t0 = time.time()
    with open(log, "a") as f:      # appended: one section per attempt
        f.write("\n===== %s | %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(cmd)))
        f.flush()
        rc = subprocess.call(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    print("    -> %s in %.1f min (log %s)" % ("ok" if rc == 0 else "FAILED rc=%d" % rc,
                                             (time.time() - t0) / 60, os.path.relpath(log, ROOT)))
    return rc == 0


def touch(path, dry=False):
    if dry:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


# parameters that change the results: a resumed run must use the same ones
RESULT_ARGS = ("keys", "K", "restarts", "attackiter", "budget", "eps", "epochs", "no_asr",
               "flip_budget", "flip_iterations")


def check_args(args):
    """Store the parameters on the first run, refuse to resume with different ones."""
    path = os.path.join(args.out, "run_args.json")
    cur = {k: getattr(args, k) for k in RESULT_ARGS}
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        diff = {k: (old.get(k), v) for k, v in cur.items() if k in old and old[k] != v and k != "keys"}
        if diff and not args.force:
            sys.exit("Parameters differ from the run stored in %s (old, new): %s\n"
                     "Re-run with the same parameters to resume, use another --out, or --force."
                     % (os.path.relpath(path, ROOT), diff))
        cur["keys"] = max(cur["keys"], old.get("keys", 0))   # adding keys later is allowed
    if not args.dry_run:
        with open(path, "w") as f:
            json.dump(cur, f, indent=1)


def step0(args):
    missing = [f"{n}_s{s}_e{C.TRAIN_EPOCHS}.pt" for n in list(C.BANK_ARCHS) + VICTIMS for s in C.TRAIN_SEEDS
               if not os.path.exists(os.path.join(C.CKPT_DIR, f"{n}_s{s}_e{C.TRAIN_EPOCHS}.pt"))]
    print("[0] checkpoints: %s" % ("all present" if not missing else "missing %s" % missing))
    logs = os.path.join(args.out, "logs")
    if missing and not run([sys.executable, "-m", "frankenstein.train_bank", "all"], ROOT,
                           os.path.join(logs, "train_bank.log"), args.dry_run):
        sys.exit("train_bank failed")
    marker = os.path.join(args.out, "done", "check_generator")
    if os.path.exists(marker):
        return
    ok = run([sys.executable, "-c", "import sys; from frankenstein import check_generator as c; "
              "sys.exit(0 if c.main() else 1)"], ROOT, os.path.join(logs, "check_generator.log"), args.dry_run)
    if not ok:
        sys.exit("generator self-test failed, see the log")
    touch(marker, args.dry_run)


# ----------------------------------------------------------------------------- Witches' Brew
def witches_runs(args):
    """(arm, surrogate args, victims) for every brew of one poison key."""
    K = args.K
    runs = [("frank", ["--net", "Frankenstein", "--ensemble", str(K)], VICTIMS),
            ("frank_q0", ["--net", "Frankenstein-q0", "--ensemble", str(K)], VICTIMS)]
    for b in C.BANK_ARCHS:
        nets = ",".join(f"{b}-s{s}" for s in C.TRAIN_SEEDS)
        runs.append((f"bank{b}", ["--net", nets, "--ensemble", str(len(C.TRAIN_SEEDS))], VICTIMS))
    for v in VICTIMS:   # surrogate = checkpoint s0, cosine read on s1, s2
        runs.append(("whitebox", ["--net", f"{v}-s0", "--ensemble", "1"], [v]))
    return runs


def step_witches(args):
    wdir = os.path.join(args.out, "witches")
    print("[1] Witches' Brew: %d keys x %d brews" % (args.keys, len(witches_runs(args))))
    for key in range(1, args.keys + 1):
        for arm, net_args, victims in witches_runs(args):
            tag = f"{arm}_{'-'.join(victims)}_k{key}" if arm == "whitebox" else f"{arm}_k{key}"
            done = os.path.join(wdir, "done", tag)
            if os.path.exists(done):
                continue
            # brewed poison cached: an interruption during the victim retraining does not re-brew;
            # --resume skips the victims already written to the table
            delta = os.path.join(wdir, "deltas", tag + ".pt")
            cmd = [sys.executable, "brew_poison.py", *net_args, "--vnet", ",".join(victims),
                   "--skip_clean_training", "--dataset", "CIFAR10",
                   "--data_path", C.DATA_DIR, "--poisonkey", str(key), "--modelkey", str(key),
                   "--budget", str(args.budget), "--eps", str(args.eps),
                   "--restarts", str(args.restarts), "--attackiter", str(args.attackiter),
                   "--epochs", str(args.epochs), "--vruns", "0" if args.no_asr else "1",
                   "--align_epochs", str(C.TRAIN_EPOCHS),
                   "--table_path", os.path.join(wdir, "tables", arm) + "/", "--name", f"_{arm}",
                   "--delta_cache", delta, "--resume"]
            if run(cmd, WITCHES, os.path.join(wdir, "logs", tag + ".log"), args.dry_run):
                touch(done, args.dry_run)


def load_witches(out):
    import pandas as pd
    frames = []
    for arm in ARMS:
        for f in glob.glob(os.path.join(out, "witches", "tables", arm, "table_*.csv")):
            df = pd.read_csv(f, sep="\t", on_bad_lines="skip")   # a row cut by an interruption
            df["arm"] = arm
            frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["arm", "victim", "poisonkey"], keep="last")
    df["dcos"] = pd.to_numeric(df["victim_cos"], errors="coerce") - pd.to_numeric(df["victim_cos_clean"], errors="coerce")
    df["success"] = pd.to_numeric(df["target_acc_reinit"], errors="coerce")
    df["clean_acc"] = pd.to_numeric(df["val_acc_reinit"], errors="coerce")
    return df


# ----------------------------------------------------------------------------- FLIP
FLIP_ARMS = {   # arm -> (expert flag, number of experts, victims)
    "frank": lambda a: ("frank-{}", a.K, VICTIMS),
    "frank_q0": lambda a: ("frank-{}-q0", a.K, VICTIMS),
    "bankA": lambda a: ("A-ckpt{}", len(C.TRAIN_SEEDS), VICTIMS),
    "bankB": lambda a: ("B-ckpt{}", len(C.TRAIN_SEEDS), VICTIMS),
}
for _v in VICTIMS:
    FLIP_ARMS["whitebox_" + _v] = (lambda v: lambda a: (f"{v}-ckpt{{}}", len(C.TRAIN_SEEDS), [v]))(_v)

FLIP_COMMON = 'trainer = "sgd"\ndataset = "cifar"\nsource_label = 9\ntarget_label = 4\npoisoner = "1xs"\n'


def write_toml(path, text, dry=False):
    if dry:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def step_flip(args):
    """Three units per arm, each with its own done marker (written only on success, so a unit
    cut by an interruption is redone): label optimisation, flip selection, one user training
    per victim."""
    fdir = os.path.join(args.out, "flip")
    done = lambda name: os.path.join(fdir, "done", name)
    print("[2] FLIP: %d label optimisations" % len(FLIP_ARMS))
    for arm, spec in FLIP_ARMS.items():
        flag, n_exp, victims = spec(args)
        adir = os.path.join(fdir, arm) + "/"
        rel = lambda d: os.path.relpath(d, os.path.join(FLIP, "experiments"))
        budget_file = adir + f"{args.flip_budget}.npy"

        if not os.path.exists(done(arm + "_labels")):
            write_toml(adir + "config.toml", f"""# generated by frankenstein.run_study
[generate_labels]
expert_model = "{flag}"
{FLIP_COMMON}output_dir = "{adir}"
lambda = 0.0

[generate_labels.expert_config]
experts = {n_exp}

[generate_labels.attack_config]
iterations = {args.flip_iterations}
one_hot_temp = 5
labels_kwargs = {{lr = 150, momentum = 0.5}}
""", args.dry_run)
            if not run([sys.executable, "run_experiment.py", rel(adir)], FLIP, adir + "labels.log", args.dry_run):
                continue
            touch(done(arm + "_labels"), args.dry_run)

        if not os.path.exists(done(arm + "_select")):
            sdir = adir + "select/"
            write_toml(sdir + "config.toml", f"""# generated by frankenstein.run_study
[select_flips]
budgets = [{args.flip_budget}]
input_label_glob = "{adir}labels.npy"
true_labels = "{adir}true.npy"
output_dir = "{adir}"
""", args.dry_run)
            if not run([sys.executable, "run_experiment.py", rel(sdir)], FLIP, sdir + "select.log", args.dry_run):
                continue
            touch(done(arm + "_select"), args.dry_run)

        for v in victims:
            if os.path.exists(done(f"{arm}_user_{v}")):
                continue
            udir = adir + f"user_{v}/"
            m1, m2 = int(0.5 * args.epochs), int(0.8 * args.epochs)
            write_toml(udir + "config.toml", f"""# generated by frankenstein.run_study
[train_user]
input_labels = "{budget_file}"
user_model = "{v}"
{FLIP_COMMON}output_dir = "{udir}"
soft = false
alpha = 0.0
epochs = {args.epochs}
scheduler_kwargs = {{milestones = [{m1}, {m2}], gamma = 0.1}}
""", args.dry_run)
            if run([sys.executable, "run_experiment.py", rel(udir)], FLIP, udir + "train_user.log", args.dry_run):
                touch(done(f"{arm}_user_{v}"), args.dry_run)


def load_flip(out):
    import pandas as pd
    rows = []
    for f in glob.glob(os.path.join(out, "flip", "*", "user_*", "paccs.npy")):
        udir = os.path.dirname(f)
        arm = os.path.basename(os.path.dirname(udir))
        victim = os.path.basename(udir)[len("user_"):]
        if not os.path.exists(os.path.join(out, "flip", "done", f"{arm}_user_{victim}")):
            continue                     # unfinished (interrupted) user training
        arm = "whitebox" if arm.startswith("whitebox_") else arm
        p = np.load(f)
        c = np.load(os.path.join(udir, "caccs.npy"))
        rows.append(dict(arm=arm, victim=victim, asr=float(p[-1][0]), clean_acc=float(c[-1][0])))
    return pd.DataFrame(rows) if rows else None


# ----------------------------------------------------------------------------- figures + summary
COLORS = {"whitebox": "#52514e", "frank": "#2a78d6", "frank_q0": "#a3a29b",
          "bankA": "#a3a29b", "bankB": "#a3a29b"}
INK, MUTED = "#0b0b0b", "#52514e"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)


def _mean_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, 0
    half = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return x.mean(), half, len(x)


def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def panel_plot(df, value, title, ylabel, path, kind="mean", zero_line=False, ylim=None):
    """One panel per victim, x = arm, dots = individual runs, bar = mean (95% CI)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(VICTIMS), figsize=(3.6 * len(VICTIMS), 3.4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, v in zip(np.atleast_1d(axes), VICTIMS):
        _style(ax)
        sub = df[df.victim == v]
        for i, arm in enumerate(ARMS):
            vals = sub.loc[sub.arm == arm, value].dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            col = COLORS[arm]
            if kind == "rate":
                m, lo, hi = _wilson(vals.sum(), len(vals))
                err = [[m - lo], [hi - m]]
                ax.text(i, hi + 0.03, "%d/%d" % (vals.sum(), len(vals)), ha="center", fontsize=7, color=MUTED)
            else:
                m, half, _ = _mean_ci(vals)
                err = [[half], [half]]
                if len(vals) > 1:
                    ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=14, color=col,
                               alpha=0.45, edgecolors="none", zorder=2)
            ax.errorbar(i, m, yerr=err, fmt="o", ms=7, color=col, ecolor=col, elinewidth=2,
                        capsize=0, zorder=3, mec="white", mew=1.5)
        if zero_line:
            ax.axhline(0, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([ARM_LABELS[a] for a in ARMS], rotation=30, ha="right")
        ax.set_title(f"Victim {v}", fontsize=10, color=INK)
        if ylim:
            ax.set_ylim(*ylim)
    np.atleast_1d(axes)[0].set_ylabel(ylabel, color=INK, fontsize=9)
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  figure", os.path.relpath(path, ROOT))


def paired(df, value, a, b):
    """Paired difference a - b over the poison keys of one victim -> (mean, ci half, n, wins)."""
    pa = df[df.arm == a].set_index("poisonkey")[value]
    pb = df[df.arm == b].set_index("poisonkey")[value]
    d = (pa - pb).dropna()
    m, h, n = _mean_ci(d.to_numpy())
    return m, h, n, int((d > 0).sum())


def step_report(args):
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    lines = ["# Frankenstein transfer study - summary", "",
             "Generated by `python -m frankenstein.run_study`. Arms: " +
             ", ".join(f"`{a}` = {ARM_LABELS[a]}" for a in ARMS) + ".",
             "Hypothesis supported on a victim if Frankenstein beats **both** wrong-architecture "
             "bank arms and the q=0 null (paired over poison keys for Witches), white-box being the ceiling.", ""]
    print("[3] figures and summary")

    wb = load_witches(args.out)
    if wb is not None and len(wb):
        panel_plot(wb, "dcos", "Witches' Brew - gradient alignment gained on clean victim checkpoints",
                   "cos(poison, target) - cos(clean, target)", os.path.join(fig_dir, "witches_alignment.png"),
                   zero_line=True)
        has_asr = wb["success"].notna().any()
        if has_asr:
            panel_plot(wb, "success", "Witches' Brew - attack success (victim retrained from scratch)",
                       "success rate (Wilson 95% CI)", os.path.join(fig_dir, "witches_asr.png"),
                       kind="rate", ylim=(-0.05, 1.15))
            panel_plot(wb, "clean_acc", "Witches' Brew - clean validation accuracy of the poisoned victim",
                       "clean validation accuracy", os.path.join(fig_dir, "witches_clean_acc.png"))
        lines += ["## Witches' Brew", "",
                  "| victim | arm | n keys | Δcos mean ± 95% CI | ASR |", "|---|---|---|---|---|"]
        for v in VICTIMS:
            for arm in ARMS:
                s = wb[(wb.victim == v) & (wb.arm == arm)]
                if not len(s):
                    continue
                m, h, n = _mean_ci(s["dcos"])
                asr = "%d/%d" % (s["success"].sum(), s["success"].notna().sum()) if has_asr else "-"
                lines.append(f"| {v} | {ARM_LABELS[arm]} | {n} | {m:+.4f} ± {h:.4f} | {asr} |")
        lines += ["", "Paired Δcos differences (Frankenstein − other arm, same poison key):", "",
                  "| victim | vs | mean ± 95% CI | Frank. wins | verdict |", "|---|---|---|---|---|"]
        for v in VICTIMS:
            sub = wb[wb.victim == v]
            beats = []
            for other in ("bankA", "bankB", "frank_q0", "whitebox"):
                m, h, n, w = paired(sub, "dcos", "frank", other)
                if n == 0:
                    continue
                sig = "Frank. better" if m - h > 0 else ("Frank. worse" if m + h < 0 else "n.s.")
                if other != "whitebox":
                    beats.append(m - h > 0)
                lines.append(f"| {v} | {ARM_LABELS[other]} | {m:+.4f} ± {h:.4f} | {w}/{n} | {sig} |")
            if beats:
                lines.append(f"| **{v}** | **all non-white-box** | | | "
                             f"**{'SUPPORTED' if all(beats) and len(beats) == 3 else 'not supported'}** |")
        lines.append("")
    else:
        lines += ["## Witches' Brew", "", "_no result yet_", ""]

    fl = load_flip(args.out)
    if fl is not None and len(fl):
        fl = fl.assign(asr=100 * fl.asr, clean_acc=100 * fl.clean_acc)
        panel_plot(fl, "asr", f"FLIP - attack success rate ({args.flip_budget} flipped labels)",
                   "ASR on triggered test images (%)", os.path.join(fig_dir, "flip_asr.png"))
        panel_plot(fl, "clean_acc", "FLIP - clean test accuracy of the user model",
                   "clean accuracy (%)", os.path.join(fig_dir, "flip_clean_acc.png"))
        lines += ["## FLIP", "", "| victim | " + " | ".join(ARM_LABELS[a] for a in ARMS) + " | verdict |",
                  "|---|" + "---|" * (len(ARMS) + 1)]
        for v in VICTIMS:
            get = {a: fl[(fl.victim == v) & (fl.arm == a)] for a in ARMS}
            cell = {a: ("%.1f (%.1f)" % (g.asr.iloc[0], g.clean_acc.iloc[0]) if len(g) else "-") for a, g in get.items()}
            others = [get[a].asr.iloc[0] for a in ("bankA", "bankB", "frank_q0") if len(get[a])]
            verdict = ("SUPPORTED" if len(get["frank"]) and len(others) == 3 and get["frank"].asr.iloc[0] > max(others)
                       else "not supported" if len(get["frank"]) and others else "-")
            lines.append(f"| {v} | " + " | ".join(cell[a] for a in ARMS) + f" | {verdict} |")
        lines += ["", "Cells: ASR % (clean accuracy %). One run per cell: differences of a few points "
                  "are within run-to-run noise.", ""]
    else:
        lines += ["## FLIP", "", "_no result yet_", ""]

    lines += ["## Figures", ""] + [f"- `figures/{os.path.basename(p)}`"
                                   for p in sorted(glob.glob(os.path.join(fig_dir, "*.png")))]
    path = os.path.join(args.out, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  summary", os.path.relpath(path, ROOT))


# ----------------------------------------------------------------------------- main
def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    attacks = set(args.attacks.split(","))
    if not args.plots_only:
        check_args(args)
        try:
            step0(args)
            if "witches" in attacks:
                step_witches(args)
            if "flip" in attacks:
                step_flip(args)
        except KeyboardInterrupt:
            print("\nInterrupted. Re-run the same command to resume where it stopped "
                  "(finished units are skipped).")
            if not args.dry_run:
                step_report(args)            # figures of what is already done
            sys.exit(130)
    if not args.dry_run:
        step_report(args)


if __name__ == "__main__":
    main()
