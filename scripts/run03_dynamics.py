"""Run 3 (long training): the faster-clock test and the fade table.

Clock test: for each seed and each AdamW evaluation step t, find the step t' at which Muon's
threshold M*(t') first equals AdamW's M*(t) (linear interpolation in log step). If Muon is
AdamW on a faster clock, t / t' is the same at every t. If AdamW levels off below Muon,
t / t' grows without bound and eventually has no answer.

Fade table, per pool size: the step of best new-task error, how far it fades by the end
(final / best, final = mean of the last 5 evaluations), and the transience step from
PLAN.md (first step after the best where new-task error is 10% above it).

usage: python scripts/run03_dynamics.py results/run03_long reports/figs/run03
"""
import math
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from analyze import COL, load, series  # noqa: E402
from stats import sign_flip_p, threshold_log2M  # noqa: E402

runs = [r for r in load(sys.argv[1]) if r["done"] and r["done"]["status"] == "ok"]
out = sys.argv[2]
A, B = "adamw", "muon"

# ---- threshold M*(t) per optimizer and seed ----
by = defaultdict(lambda: defaultdict(dict))  # opt -> seed -> M -> {step: alpha}
for r in runs:
    c = r["cfg"]
    by[c["opt"]][c["seed"]][c["M"]] = {row["step"]: row["alpha_ood"] for row in r["rows"]}
seeds = sorted(set(by[A]) & set(by[B]))
Ms = sorted(by[A][seeds[0]])
steps = sorted(by[A][seeds[0]][Ms[0]])


def mstar_curve(opt, sd):
    return np.array([threshold_log2M(Ms, [by[opt][sd][M][s] for M in Ms]) for s in steps])


curves = {(o, sd): mstar_curve(o, sd) for o in (A, B) for sd in seeds}
logt = np.log10(np.maximum(steps, 1))


def first_reach(curve, level):
    """log10 step where curve first reaches level (interpolated), or nan. Steps below 1,000 are
    skipped: alpha means nothing before the model is near either reference predictor."""
    for i in range(1, len(curve)):
        if steps[i - 1] < 1000:
            continue
        a0, a1 = curve[i - 1], curve[i]
        if np.isfinite(a0) and np.isfinite(a1) and a0 < level <= a1:
            return logt[i - 1] + (level - a0) / (a1 - a0) * (logt[i] - logt[i - 1])
    return math.nan


# AdamW's curve is below the grid floor (32 tasks) until about step 10k; use steps >= 15k.
probe = [s for s in steps if s in (15000, 21000, 30000, 45000, 60000, 90000, 120000, 150000, 210000, 255000, 300000)]
print("## Clock test: steps Muon needs to match AdamW's threshold at step t\n")
print("| AdamW step t | AdamW log2 M* | Muon matches at step | ratio t / t' (per seed) | mean ratio |")
print("|---|---|---|---|---|")
ratio_rows = []
for s in probe:
    i = steps.index(s)
    rat, tp, lv = [], [], []
    for sd in seeds:
        level = curves[(A, sd)][i]
        lt = first_reach(curves[(B, sd)], level)
        lv.append(level)
        tp.append(10 ** lt if np.isfinite(lt) else math.nan)
        rat.append(s / 10 ** lt if np.isfinite(lt) else math.nan)
    ratio_rows.append((s, rat))
    print(f"| {s:,} | {np.nanmean(lv):.2f} | {np.nanmean(tp):,.0f} | {' '.join(f'{x:.1f}' for x in rat)} | {np.nanmean(rat):.1f} |")

# slope of M*(t) per decade of steps over the last decade (30k -> 300k)
i30, i300 = steps.index(30000), steps.index(300000)
i150 = steps.index(150000)
print("\n## Is either threshold levelling off?\n")
print("| optimizer | log2 M* at 30k | at 150k | at 300k | rise 30k->300k (doublings per 10x steps) | rise 150k->300k |")
print("|---|---|---|---|---|---|")
for o in (A, B):
    v30 = np.mean([curves[(o, sd)][i30] for sd in seeds])
    v150 = np.mean([curves[(o, sd)][i150] for sd in seeds])
    v300 = np.mean([curves[(o, sd)][i300] for sd in seeds])
    print(f"| {o} | {v30:.2f} | {v150:.2f} | {v300:.2f} | {v300 - v30:+.2f} | {v300 - v150:+.2f} |")

# figure: M*(t) for both, and AdamW shifted left by the mean ratio
mean_ratio = float(np.nanmean([x for _, r in ratio_rows for x in r]))
fig, ax = plt.subplots(figsize=(6.5, 4))
for o in (A, B):
    mu = np.nanmean([curves[(o, sd)] for sd in seeds], axis=0)
    ax.plot(steps, mu, color=COL[o], label=o)
mu_a = np.nanmean([curves[(A, sd)] for sd in seeds], axis=0)
ax.plot(np.array(steps) / mean_ratio, mu_a, color=COL[A], ls=":", label=f"adamw, steps / {mean_ratio:.1f}")
ax.set_xscale("log")
ax.set_xlim(1000, 3.2e5)
ax.set_xlabel("training step")
ax.set_ylabel("log2 M* (pool size where alpha = 0.5)")
ax.set_title(f"Run 3: threshold over 300k steps ({len(seeds)} seeds; grid floor 32 tasks)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out + "/clock_test.png", dpi=130)
plt.close(fig)


# ---- fade table ----
from scripts.run03_fade import fade_stats  # noqa: E402


fs = {(r["cfg"]["opt"], r["cfg"]["M"], r["cfg"]["seed"]): fade_stats(r) for r in runs}
print("\n## Fade, per pool size (means over seeds; errors relative to ridge)\n")
print("| M | best new-task err, AdamW / Muon | step of best, AdamW / Muon | 10%-fade step, AdamW / Muon "
      "| final new-task err, AdamW / Muon | final / best, AdamW / Muon | final / at 150k, AdamW / Muon |")
print("|---|---|---|---|---|---|---|")
for M in Ms:
    g = {o: [fs[(o, M, sd)] for sd in seeds] for o in (A, B)}
    m = lambda o, k: np.mean([x[k] for x in g[o]])  # noqa: E731
    gm = lambda o, k: 10 ** np.mean([np.log10(x[k]) for x in g[o]])  # noqa: E731
    print(f"| {M} | {m(A,'best'):.2f}x / {m(B,'best'):.2f}x | {gm(A,'best_step'):,.0f} / {gm(B,'best_step'):,.0f} | "
          f"{gm(A,'t10'):,.0f} / {gm(B,'t10'):,.0f} | {m(A,'final'):.2f}x / {m(B,'final'):.2f}x | "
          f"{m(A,'fade'):.2f} / {m(B,'fade'):.2f} | {m(A,'still'):.2f} / {m(B,'still'):.2f} |")

print(f"\n## Paired, Muon minus AdamW ({len(seeds)} seeds; smallest possible p = {2 / 2 ** len(seeds):.3f})\n")
print("| M | log2 (step of best) | seeds Muon earlier | log2 (10%-fade step) | seeds Muon earlier | log2 (final/best) | seeds Muon fades more |")
print("|---|---|---|---|---|---|---|")
for M in Ms:
    d1 = [math.log2(fs[(B, M, sd)]["best_step"] / fs[(A, M, sd)]["best_step"]) for sd in seeds]
    d2 = [math.log2(fs[(B, M, sd)]["t10"] / fs[(A, M, sd)]["t10"]) for sd in seeds]
    d3 = [math.log2(fs[(B, M, sd)]["fade"] / fs[(A, M, sd)]["fade"]) for sd in seeds]
    print(f"| {M} | {np.mean(d1):+.2f} (p={sign_flip_p(d1)[0]:.3f}) | {sum(x < 0 for x in d1)}/{len(seeds)} | "
          f"{np.mean(d2):+.2f} (p={sign_flip_p(d2)[0]:.3f}) | {sum(x < 0 for x in d2)}/{len(seeds)} | "
          f"{np.mean(d3):+.2f} (p={sign_flip_p(d3)[0]:.3f}) | {sum(x > 0 for x in d3)}/{len(seeds)} |")
