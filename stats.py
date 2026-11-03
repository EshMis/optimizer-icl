"""Primary endpoints of PLAN.md, paired across seeds, plus mechanism figures.

usage: python stats.py results/run02_main --figs reports/figs/run02 [--a adamw --b muon]
Endpoint values per run use the mean of the last 5 evaluations, so one noisy eval cannot
decide them. Pairs are (optimizer a, optimizer b) runs with the same seed and pool size.
"""
import argparse
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import COL, load, onset_step, series


def tail_mean(r, key, n=5):
    return float(np.mean([row[key] for row in r["rows"][-n:]]))


def threshold_log2M(Ms, alphas):
    """log2 M where alpha crosses 0.5, by linear interpolation in log2 M (first downward crossing)."""
    x = np.log2(Ms)
    for i in range(len(x) - 1):
        a0, a1 = alphas[i], alphas[i + 1]
        if a0 >= 0.5 > a1:
            return x[i] + (a0 - 0.5) / (a0 - a1) * (x[i + 1] - x[i])
    return math.nan


def sign_flip_p(d, n_perm=100_000, seed=0):
    """Two-sided paired sign-flip permutation p-value for mean(d) != 0 (exact when 2^n <= n_perm)."""
    d = np.asarray([v for v in d if np.isfinite(v)])
    n = len(d)
    if n == 0:
        return math.nan, 0
    obs = abs(d.mean())
    if 2 ** n <= n_perm:
        signs = np.array([[1 if (i >> j) & 1 else -1 for j in range(n)] for i in range(2 ** n)])
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_perm, n))
    return float((np.abs((signs * d).mean(1)) >= obs - 1e-12).mean()), n


def boot_ci(d, n=20_000, seed=0):
    d = np.asarray([v for v in d if np.isfinite(v)])
    if len(d) < 2:
        return math.nan, math.nan
    m = np.random.default_rng(seed).choice(d, size=(n, len(d))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def report_paired(name, d, unit=""):
    p, n = sign_flip_p(d)
    lo, hi = boot_ci(d)
    d = [v for v in d if np.isfinite(v)]
    mean = np.mean(d) if d else math.nan
    return f"| {name} | {n} | {mean:+.3f}{unit} | [{lo:+.3f}, {hi:+.3f}] | {p:.3f} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--figs", required=True)
    ap.add_argument("--a", default="adamw")
    ap.add_argument("--b", default="muon")
    ap.add_argument("--lr_a", type=float, help="keep only optimizer-a runs at this matrix LR")
    ap.add_argument("--lr_b", type=float, help="keep only optimizer-b runs at this matrix LR")
    args = ap.parse_args()
    os.makedirs(args.figs, exist_ok=True)
    runs = [r for r in load(args.results) if r["done"] is not None and r["done"]["status"] == "ok"]
    for o, lr in ((args.a, args.lr_a), (args.b, args.lr_b)):
        if lr is not None:
            runs = [r for r in runs if r["cfg"]["opt"] != o or math.isclose(r["cfg"]["lr"], lr)]
    keys = [(r["cfg"]["opt"], r["cfg"]["M"], r["cfg"]["seed"]) for r in runs]
    assert len(keys) == len(set(keys)), "several runs per (optimizer, M, seed): pass --lr_a/--lr_b"
    by = {(r["cfg"]["opt"], r["cfg"]["M"], r["cfg"]["seed"]): r for r in runs}
    opts = [args.a, args.b]
    Ms = sorted({k[1] for k in by if k[1] > 0})
    seeds = sorted({k[2] for k in by})

    # 1. task-diversity threshold per (optimizer, seed)
    thr = defaultdict(dict)
    for o in opts:
        for s in seeds:
            ms = [M for M in Ms if (o, M, s) in by]
            if len(ms) >= 3:
                thr[o][s] = threshold_log2M(ms, [tail_mean(by[(o, M, s)], "alpha_ood") for M in ms])
    lines = ["| endpoint (b minus a) | pairs | mean diff | 95% CI | p (sign-flip) |", "|---|---|---|---|---|"]
    d_thr = [thr[args.b][s] - thr[args.a][s] for s in seeds if s in thr[args.a] and s in thr[args.b]]
    lines.append(report_paired("log2 M* (task-diversity threshold)", d_thr))

    # 2. onset, 3. final errors, per pool size
    for M in sorted({k[1] for k in by}, key=lambda m: m or 1 << 40):
        pairs = [(by[(args.a, M, s)], by[(args.b, M, s)]) for s in seeds
                 if (args.a, M, s) in by and (args.b, M, s) in by]
        if not pairs:
            continue
        Ml = "inf" if M == 0 else M
        on = [math.log2(onset_step(rb) / onset_step(ra)) if onset_step(ra) and onset_step(rb) else math.nan
              for ra, rb in pairs]
        lines.append(report_paired(f"M={Ml}: log2 onset-step ratio", on))
        for key in ("mse_ood_last", "mse_id_last", "alpha_ood"):
            d = [tail_mean(rb, key) - tail_mean(ra, key) for ra, rb in pairs]
            lines.append(report_paired(f"M={Ml}: {key}", d))
    thr_line = ", ".join(f"{o}: " + " ".join(f"{v:.2f}" for v in thr[o].values()) for o in opts)
    out = "\n".join(lines) + f"\n\nper-seed log2 M*: {thr_line}\n"
    print(out)
    open(os.path.join(args.figs, "stats.md"), "w").write(out)

    # figure: alpha and errors vs M (mean +- s.e. over seeds)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for o in opts:
        for ax, key, lab in zip(axes, ("alpha_ood", "mse_ood_last", "mse_id_last"),
                                ("alpha on new tasks", "new-task error / ridge", "pool-task error / dMMSE")):
            mu, se = [], []
            for M in Ms:
                v = []
                for s in seeds:
                    r = by.get((o, M, s))
                    if r is None:
                        continue
                    x = tail_mean(r, key)
                    if key == "mse_ood_last":
                        x /= r["base"]["ridge_ood_mse_pos"][-1]
                    if key == "mse_id_last":
                        x /= r["base"]["dmmse_id_mse_pos"][-1]
                    v.append(x)
                mu.append(np.mean(v) if v else np.nan)
                se.append(np.std(v) / math.sqrt(len(v)) if len(v) > 1 else 0)
            ax.errorbar(Ms, mu, yerr=se, color=COL[o], marker="o", ms=3, capsize=2, label=o)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("task pool size M")
            ax.set_title(lab)
    axes[0].axhline(0.5, color="k", ls=":", lw=0.8)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.figs, "vs_pool_size.png"), dpi=130)
    plt.close(fig)

    # figure: mechanism traces at each M (seed-mean): effective rank of weight change, attention to labels
    show = Ms[:: max(1, len(Ms) // 4)][:4] + [0] if 0 in {k[1] for k in by} else Ms[:: max(1, len(Ms) // 4)][:4]
    fig, axes = plt.subplots(3, len(show), figsize=(3.6 * len(show), 8), squeeze=False)
    for j, M in enumerate(show):
        for o in opts:
            rs = [by[(o, M, s)] for s in seeds if (o, M, s) in by]
            if not rs:
                continue
            for r in rs:
                st = np.array([row["step"] for row in r["rows"]])
                er_attn = [np.mean([v[4] for k, v in row["spec"].items() if k.split(".")[1] in "qkvo"]) for row in r["rows"]]
                er_mlp = [np.mean([v[4] for k, v in row["spec"].items() if k.split(".")[1] in ("fc", "proj")]) for row in r["rows"]]
                ym = [np.mean(row["attn_ymass"]) for row in r["rows"]]
                for i, y in enumerate((er_attn, er_mlp, ym)):
                    axes[i][j].plot(np.maximum(st, 1), y, color=COL[o], alpha=0.5, lw=0.8)
        axes[0][j].set_title(f"M = {'inf' if M == 0 else M}")
        for i in range(3):
            axes[i][j].set_xscale("log")
        axes[2][j].set_xlabel("step")
    axes[0][0].set_ylabel("eff. rank of ΔW, attention")
    axes[1][0].set_ylabel("eff. rank of ΔW, MLP")
    axes[2][0].set_ylabel("attention mass on y tokens")
    fig.tight_layout()
    fig.savefig(os.path.join(args.figs, "mechanism.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
