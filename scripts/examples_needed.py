"""Proposal question 3: how many in-context examples does each model need? (PLAN.md endpoint 4)

Per run, from new-task error at each context length k = 0..15 (k = examples seen), averaged
over the last 5 evaluations (stats.py convention), against ridge on the same prompts:

- k_match: the smallest k from which the model stays within 10% of ridge for every later k.
  PLAN.md says "the smallest k where error is within 10% of ridge at the same k"; read
  literally that is always k = 0, where every predictor outputs about 0 and all errors are
  about 8.1, so we require it to hold from k onwards. 16 = never (not even at k = 15).
- excess: the area between the model's error-vs-k curve and ridge's, as mean over k of
  (model - ridge). In the same units as the error; 0 = optimal at every context length.
- k_equiv (added after the plan, for reading only): how many examples ridge needs to reach
  the model's error at k = 15. 15 = as good as optimal; interpolated between positions.

usage:
  python scripts/examples_needed.py final results/run02_main reports/figs/run02
  python scripts/examples_needed.py over_time results/run03_long reports/figs/run03
"""
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from analyze import COL, load  # noqa: E402
from stats import boot_ci, sign_flip_p  # noqa: E402

TOL = 1.10


def measures(err, ridge):
    err, ridge = np.asarray(err, float), np.asarray(ridge, float)
    ok = err <= TOL * ridge
    k_match = len(err)
    for k in range(len(err) - 1, -1, -1):
        if not ok[k]:
            break
        k_match = k
    excess = float(np.mean(err - ridge))
    # ridge error falls with k (non-increasing in expectation); find first k where it is <= model's last error
    target = err[-1]
    k_equiv = float(len(err) - 1)
    for k in range(1, len(ridge)):
        if ridge[k] <= target:
            k_equiv = k - 1 + (ridge[k - 1] - target) / (ridge[k - 1] - ridge[k]) if ridge[k - 1] > ridge[k] else float(k)
            break
    return k_match, excess, min(k_equiv, len(err) - 1)


def at(r, idx):
    """Mean new-task error per position over evaluations idx (a slice of rows)."""
    return np.mean([row["mse_ood_pos"] for row in r["rows"][idx]], axis=0)


def label_M(M):
    return "inf" if M == 0 else str(M)


def final(results, figs):
    runs = [r for r in load(results) if r["done"] and r["done"]["status"] == "ok"]
    res = {}
    for r in runs:
        c = r["cfg"]
        res[(c["opt"], c["lr"], c["M"], c["seed"])] = measures(at(r, slice(-5, None)), r["base"]["ridge_ood_mse_pos"])
    Ms = sorted({k[2] for k in res}, key=lambda m: m or 1 << 40)
    settings = sorted({(k[0], k[1]) for k in res}, key=lambda s: (["sgdm", "adamw", "signum", "muon"].index(s[0]), s[1]))

    print("## Means over seeds, at the end of training\n")
    print("k_match: examples after which the model stays within 10% of ridge (16 = never). "
          "excess: mean over k of (model - ridge) error. k_equiv: ridge needs this many examples "
          "to match the model's error with 15.\n")
    print("| pool size | " + " | ".join(f"{o} {lr:g}" for o, lr in settings) + " |")
    print("|---|" + "---|" * len(settings))
    for name, j, fmt in (("k_match", 0, "{:.1f}"), ("excess", 1, "{:.2f}"), ("k_equiv", 2, "{:.1f}")):
        print(f"| **{name}** |" + " |" * len(settings))
        for M in Ms:
            cells = []
            for o, lr in settings:
                v = [res[(o, lr, M, s)][j] for s in range(6) if (o, lr, M, s) in res]
                cells.append(fmt.format(np.mean(v)) if v else "")
            print(f"| {label_M(M)} | " + " | ".join(cells) + " |")

    print("\n## Paired, Muon minus AdamW, per learning-rate pairing (6 seeds; smallest p = 0.031)\n")
    print("| pairing | pool size | k_match diff | seeds Muon needs more | excess diff [95% CI] | p | seeds Muon worse | k_equiv diff |")
    print("|---|---|---|---|---|---|---|---|")
    lrs_a = sorted({lr for o, lr in settings if o == "adamw"})
    lrs_b = sorted({lr for o, lr in settings if o == "muon"})
    for la in lrs_a:
        for lb in lrs_b:
            for M in Ms:
                pairs = [(res[("adamw", la, M, s)], res[("muon", lb, M, s)]) for s in range(6)
                         if ("adamw", la, M, s) in res and ("muon", lb, M, s) in res]
                dk = [b[0] - a[0] for a, b in pairs]
                de = [b[1] - a[1] for a, b in pairs]
                dq = [b[2] - a[2] for a, b in pairs]
                lo, hi = boot_ci(de)
                print(f"| AdamW {la:g} / Muon {lb:g} | {label_M(M)} | {np.mean(dk):+.1f} | {sum(x > 0 for x in dk)}/{len(dk)} "
                      f"| {np.mean(de):+.3f} [{lo:+.3f}, {hi:+.3f}] | {sign_flip_p(de)[0]:.3f} | {sum(x > 0 for x in de)}/{len(de)} "
                      f"| {np.mean(dq):+.1f} |")

    # figure: error vs k relative to ridge, per pool size, AdamW 1e-4 vs Muon 0.005
    show = [M for M in Ms if M in (32, 128, 512, 2048, 0)]
    fig, axes = plt.subplots(1, len(show), figsize=(3.4 * len(show), 3.2), sharey=True)
    for ax, M in zip(axes, show):
        for o, lr in (("adamw", 1e-4), ("muon", 0.005)):
            rr = [r for r in runs if (r["cfg"]["opt"], r["cfg"]["lr"], r["cfg"]["M"]) == (o, lr, M)]
            ratio = np.mean([at(r, slice(-5, None)) / np.asarray(r["base"]["ridge_ood_mse_pos"]) for r in rr], axis=0)
            ax.plot(range(16), ratio, color=COL[o], label=f"{o} {lr:g}")
        ax.axhline(1, color="k", lw=0.8)
        ax.axhline(TOL, color="k", lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_title(f"M = {label_M(M)}")
        ax.set_xlabel("examples in the prompt (k)")
    axes[0].set_ylabel("new-task error / ridge")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs + "/examples_needed.png", dpi=130)


def over_time(results, figs):
    runs = [r for r in load(results) if r["done"] and r["done"]["status"] == "ok"]
    Ms = sorted({r["cfg"]["M"] for r in runs})
    opts = sorted({r["cfg"]["opt"] for r in runs})
    steps = [row["step"] for row in runs[0]["rows"]]
    show = [s for s in (1000, 3000, 10000, 30000, 100000, 300000) if s in steps]
    print("## k_equiv over training: ridge needs this many examples to match the model with 15 (mean over seeds; 15 = optimal)\n")
    print("| pool size | optimizer | " + " | ".join(f"{s:,}" for s in show) + " | best (step) |")
    print("|---|---|" + "---|" * (len(show) + 1))
    curves = defaultdict(list)
    for M in Ms:
        for o in opts:
            rr = [r for r in runs if (r["cfg"]["opt"], r["cfg"]["M"]) == (o, M)]
            per = []
            for r in rr:
                ridge = r["base"]["ridge_ood_mse_pos"]
                ks = []
                for i in range(len(r["rows"])):
                    lo = max(0, i - 1)
                    ks.append(measures(at(r, slice(lo, i + 2)), ridge)[2])  # 3-eval window
                per.append(ks)
            mean = np.mean(per, axis=0)
            curves[(o, M)] = mean
            ib = int(np.argmax(mean))
            print(f"| {M} | {o} | " + " | ".join(f"{mean[steps.index(s)]:.1f}" for s in show) + f" | {mean[ib]:.1f} ({steps[ib]:,}) |")
    fig, axes = plt.subplots(1, len(Ms), figsize=(3.4 * len(Ms), 3.2), sharey=True)
    for ax, M in zip(axes, Ms):
        for o in opts:
            ax.plot(np.maximum(steps, 1), curves[(o, M)], color=COL[o], label=o)
        ax.set_xscale("log")
        ax.set_xlim(100, 3.2e5)
        ax.set_title(f"M = {M}")
        ax.set_xlabel("step")
    axes[0].set_ylabel("ridge-equivalent examples (of 15)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs + "/examples_needed_over_time.png", dpi=130)


if __name__ == "__main__":
    {"final": final, "over_time": over_time}[sys.argv[1]](sys.argv[2], sys.argv[3])
