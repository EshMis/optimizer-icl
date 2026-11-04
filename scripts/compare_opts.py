"""All optimizers on one set of axes: threshold per seed and errors vs pool size.

usage: python scripts/compare_opts.py results/run02_main --figs reports/figs/run04 \
           --runs adamw:1e-4 adamw:3e-4 sgdm:0.1 signum:1e-4 muon:0.005 muon:0.01
Prints a markdown table (threshold M* per setting, then mean errors per pool size) and
writes vs_pool_size_all.png. Endpoint values are the mean of the last 5 evaluations.
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze import COL, load  # noqa: E402
from stats import tail_mean, threshold_log2M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--figs", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="opt:lr")
    a = ap.parse_args()
    os.makedirs(a.figs, exist_ok=True)
    runs = [r for r in load(a.results) if r["done"] is not None and r["done"]["status"] == "ok"]
    settings = [(s.split(":")[0], float(s.split(":")[1])) for s in a.runs]
    by = {}
    for r in runs:
        for o, lr in settings:
            if r["cfg"]["opt"] == o and math.isclose(r["cfg"]["lr"], lr):
                by[(o, lr, r["cfg"]["M"], r["cfg"]["seed"])] = r
    Ms = sorted({k[2] for k in by if k[2] > 0})
    seeds = sorted({k[3] for k in by})

    print("| setting | threshold M* (mean over seeds) | log2 M* per seed |")
    print("|---|---|---|")
    for o, lr in settings:
        t = [threshold_log2M(Ms, [tail_mean(by[(o, lr, M, s)], "alpha_ood") for M in Ms]) for s in seeds]
        print(f"| {o} {lr:g} | {2 ** np.nanmean(t):.0f} | {' '.join(f'{v:.2f}' for v in t)} |")

    rel = {}
    for o, lr in settings:
        for M in Ms + [0]:
            vo, vi = [], []
            for s in seeds:
                r = by.get((o, lr, M, s))
                if r is None:
                    continue
                vo.append(tail_mean(r, "mse_ood_last") / r["base"]["ridge_ood_mse_pos"][-1])
                vi.append(tail_mean(r, "mse_id_last") / r["base"]["dmmse_id_mse_pos"][-1])
            rel[(o, lr, M)] = (vo, vi)
    print("\nError on new tasks vs ridge / error on pool tasks vs dMMSE (seed mean; median in brackets when it differs by >10%)\n")
    print("| M | " + " | ".join(f"{o} {lr:g}" for o, lr in settings) + " |")
    print("|---|" + "---|" * len(settings))
    for M in Ms + [0]:
        cells = []
        for o, lr in settings:
            vo, vi = rel[(o, lr, M)]
            mo, md = np.mean(vo), np.median(vo)
            extra = f" [{md:.2f}]" if abs(mo - md) > 0.1 * md else ""
            cells.append(f"{mo:.2f}x{extra} / {np.mean(vi):.2f}x")
        print(f"| {'inf' if M == 0 else M} | " + " | ".join(cells) + " |")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for o, lr in settings:
        same = [x for x in settings if x[0] == o]
        ls = "-" if same.index((o, lr)) == 0 else "--"
        for ax, key in zip(axes, ("alpha_ood", "ood", "id")):
            mu, se = [], []
            for M in Ms:
                if key == "alpha_ood":
                    v = [tail_mean(by[(o, lr, M, s)], "alpha_ood") for s in seeds if (o, lr, M, s) in by]
                else:
                    v = rel[(o, lr, M)][0 if key == "ood" else 1]
                mu.append(np.median(v))
                se.append(np.std(v) / math.sqrt(len(v)))
            ax.errorbar(Ms, mu, yerr=se, color=COL[o], ls=ls, marker="o", ms=3, capsize=2, label=f"{o} {lr:g}")
    for ax, t in zip(axes, ("alpha on new tasks (1 = memorising)", "new-task error / ridge", "pool-task error / dMMSE")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("task pool size M")
        ax.set_title(t)
    axes[0].axhline(0.5, color="k", ls=":", lw=0.8)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.figs, "vs_pool_size_all.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
