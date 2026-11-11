"""Task-diversity threshold M*(t) at every evaluation step, per optimizer setting.

If Muon were AdamW on a faster clock, its M*(t) curve would be AdamW's shifted left on a
log-step axis by a constant (the onset ratio). The figure and table test that directly.
usage: python scripts/threshold_over_time.py results/run02_main reports/figs/run02
"""
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from analyze import COL, load  # noqa: E402
from stats import threshold_log2M  # noqa: E402

runs = [r for r in load(sys.argv[1]) if r["cfg"]["M"] > 0]
groups = defaultdict(lambda: defaultdict(dict))   # (opt, lr) -> seed -> M -> {step: alpha}
for r in runs:
    c = r["cfg"]
    groups[(c["opt"], c["lr"])][c["seed"]][c["M"]] = {row["step"]: row["alpha_ood"] for row in r["rows"]}
first = next(iter(next(iter(groups.values())).values()))
steps = sorted(first[min(first)].keys())
steps = [s for s in steps if s >= 300]
fig, ax = plt.subplots(figsize=(6.5, 4))
summary = {}
for (o, lr), seeds in sorted(groups.items()):
    curve = []
    for s in steps:
        v = []
        for sd, byM in seeds.items():
            Ms = sorted(byM)
            v.append(threshold_log2M(Ms, [byM[M][s] for M in Ms]))
        curve.append((np.nanmean(v), np.nanstd(v) / np.sqrt(np.sum(np.isfinite(v)))))
    mu = np.array([c[0] for c in curve])
    se = np.array([c[1] for c in curve])
    summary[(o, lr)] = dict(zip(steps, mu))
    ls = "-" if lr in (1e-4, 0.005) else "--"
    ax.plot(steps, mu, color=COL[o], ls=ls, label=f"{o} lr={lr:g}")
    ax.fill_between(steps, mu - se, mu + se, color=COL[o], alpha=0.15)
ax.set_xscale("log")
ax.set_xlabel("training step")
ax.set_ylabel("log2 M* (pool size where alpha = 0.5)")
n_seeds = len(next(iter(groups.values())))
ax.set_title(f"Task-diversity threshold over training (mean ± s.e., {n_seeds} seeds)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(sys.argv[2] + "/threshold_over_time.png", dpi=130)
show = [s for s in steps if s in (300, 1000, 3000, 6000, 9000, 15000, 21000, 30000, 60000, 105000, 150000, 210000, 300000)] or steps[::10]
print("step " + " ".join(f"{s:>7}" for s in show))
for k, d in summary.items():
    print(f"{k[0]}:{k[1]:g}".ljust(13) + " ".join(f"{d[s]:7.2f}" for s in show))
