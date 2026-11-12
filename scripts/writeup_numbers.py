"""Figures quoted in WRITEUP.md that the per-run reports give only in log2 form or not at all.

usage: python scripts/writeup_numbers.py results > reports/writeup_numbers.md

Everything is recomputed from the saved metrics with the same functions the reports use. A
"doubling" is a factor of 2 in the threshold M*, so d doublings is a factor of 2^d.
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze import load, onset_step  # noqa: E402
from stats import tail_mean, threshold_log2M  # noqa: E402
from scripts.run03_fade import fade_stats  # noqa: E402

root = sys.argv[1]
ok = lambda r: r["done"] is not None and r["done"]["status"] == "ok"  # noqa: E731

print("# Derived figures used in WRITEUP.md\n")
print("Written by `scripts/writeup_numbers.py`. Do not edit by hand.\n")

# ---- 1. how many runs, and what they cost ----
print("## Runs and compute\n")
print("| Results directory | Steps per run | Finished runs | Wall-clock hours | CPU cores per run | Core-hours |")
print("|---|---|---|---|---|---|")
n_all, ch_all = 0, 0.0
for d in sorted(glob.glob(os.path.join(root, "run*"))):
    by = defaultdict(lambda: [0, 0.0])
    for f in glob.glob(os.path.join(d, "*", "done.json")):
        dn = json.load(open(f))
        if dn["status"] != "ok":
            continue
        c = json.load(open(os.path.join(os.path.dirname(f), "config.json")))
        by[c["steps"]][0] += 1
        by[c["steps"]][1] += dn["seconds"] / 3600
    cores = 16 if os.path.basename(d) == "run01_calibrate" else 8
    for steps, (n, h) in sorted(by.items()):
        print(f"| {os.path.basename(d)} | {steps:,} | {n} | {h:,.0f} | {cores} | {h * cores:,.0f} |")
        n_all += n
        ch_all += h * cores
print(f"\nTotal: {n_all} finished training runs, about {round(ch_all, -2):,.0f} core-hours.\n")
mins = sorted(json.load(open(f))["seconds"] / 60 for d in ("run02_main", "run05b_wd")
              for f in glob.glob(os.path.join(root, d, "*", "done.json")) if json.load(open(f))["status"] == "ok")
print(f"A 30,000-step run on 8 cores took a median of {np.median(mins):.0f} minutes "
      f"(mean {np.mean(mins):.0f}, 90th percentile {np.percentile(mins, 90):.0f}; {len(mins)} runs).\n")

# ---- 1b. runs as designed: one line per run in tasks/ ----
print("## Runs as designed (lines in `tasks/`)\n")
tdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks")
n_des, n5 = 0, 0
for f in sorted(glob.glob(os.path.join(tdir, "*.txt"))):
    n = sum(1 for line in open(f) if line.strip() and not line.startswith("#"))
    n_des += n
    n5 += n if os.path.basename(f).startswith("run05") else 0
    print(f"- {os.path.basename(f)}: {n}")
print(f"\nDesigned in all: {n_des} runs, of which Run 5 is {n5}.\n")

runs2 = [r for r in load(os.path.join(root, "run02_main")) if ok(r)]
cell = defaultdict(dict)
for r in runs2:
    c = r["cfg"]
    if c["wd"] == 0:
        cell[(c["opt"], round(math.log10(c["lr"]), 3))][(c["M"], c["seed"])] = r
K = lambda o, lr: (o, round(math.log10(lr), 3))  # noqa: E731
GRID = [16, 32, 64, 128, 256, 512, 1024]
A_LR, M_LR = [3e-5, 1e-4, 3e-4, 1e-3], [0.0025, 0.005, 0.01]


def thr(o, lr):
    c = cell[K(o, lr)]
    return np.array([threshold_log2M(GRID, [tail_mean(c[(M, s)], "alpha_ood") for M in GRID]) for s in range(6)])


# ---- 2. the range of the gap over stable learning rates ----
print("## Threshold gap over the learning rates that train at every pool size (30,000 steps)\n")
gaps = {(la, lm): float(np.mean(thr("muon", lm) - thr("adamw", la))) for la in A_LR for lm in M_LR}
allpos = all(bool(np.all(thr("muon", lm) - thr("adamw", la) > 0)) for la in A_LR for lm in M_LR)
lo, hi = min(gaps.values()), max(gaps.values())
print(f"- {len(gaps)} pairings (4 AdamW x 3 Muon learning rates). Muon above AdamW in 6 of 6 seeds in every pairing: {allpos}.")
print(f"- Smallest gap {lo:.2f} doublings = {2 ** lo:.2f}x more tasks; largest {hi:.2f} doublings = {2 ** hi:.2f}x.")
ta, tm = [2 ** thr('adamw', lr).mean() for lr in A_LR], [2 ** thr('muon', lr).mean() for lr in M_LR]
print(f"- AdamW thresholds {min(ta):.0f} to {max(ta):.0f} tasks; Muon thresholds {min(tm):.0f} to {max(tm):.0f} tasks.")
print(f"- AdamW 3e-4 against Muon 0.005: {2 ** thr('adamw', 3e-4).mean():.0f} against {2 ** thr('muon', 0.005).mean():.0f} tasks, "
      f"{gaps[(3e-4, 0.005)]:.2f} doublings = {2 ** gaps[(3e-4, 0.005)]:.2f}x.\n")

# ---- 3. onset, Run 2 pairings ----
print("## ICL onset in Run 2 (pool sizes 32 to infinite; AdamW 1e-4 or 3e-4 against Muon 0.005 or 0.01)\n")
rat = []
for la in (1e-4, 3e-4):
    for lm in (0.005, 0.01):
        for M in [32, 64, 128, 256, 512, 1024, 2048, 4096, 0]:
            v = [math.log2(onset_step(cell[K("adamw", la)][(M, s)]) / onset_step(cell[K("muon", lm)][(M, s)])) for s in range(6)]
            rat.append((float(np.mean(v)), all(x > 0 for x in v)))
print(f"- Muon's onset is sooner by a factor of {2 ** min(r[0] for r in rat):.1f}x to {2 ** max(r[0] for r in rat):.1f}x "
      f"(geometric mean over 6 seeds, per pairing and pool size; {len(rat)} cells). "
      f"Muon sooner in 6 of 6 seeds in {sum(r[1] for r in rat)} of {len(rat)} cells.\n")

# ---- 4. Run 3 ----
runs3 = [r for r in load(os.path.join(root, "run03_long")) if ok(r)]
by = defaultdict(lambda: defaultdict(dict))
for r in runs3:
    c = r["cfg"]
    by[c["opt"]][c["seed"]][c["M"]] = {row["step"]: row["alpha_ood"] for row in r["rows"]}
seeds = sorted(by["adamw"])
Ms = sorted(by["adamw"][seeds[0]])
print("## Long runs (Run 3: 300,000 steps, AdamW 3e-4 against Muon 0.005)\n")
print(f"- {len(runs3)} runs: pool sizes {', '.join(str(m) for m in Ms)}; {len(seeds)} seeds.")
for step in (30000, 300000):
    t = {o: np.array([threshold_log2M(Ms, [by[o][sd][M][step] for M in Ms]) for sd in seeds]) for o in ("adamw", "muon")}
    d = t["muon"] - t["adamw"]
    print(f"- At step {step:,}: AdamW's threshold is {2 ** t['adamw'].mean():.0f} tasks and Muon's {2 ** t['muon'].mean():.0f} "
          f"({d.mean():.2f} doublings, Muon above in {(d > 0).sum()} of {len(d)} seeds).")
fs = {(r["cfg"]["opt"], r["cfg"]["M"], r["cfg"]["seed"]): fade_stats(r) for r in runs3}
pk = [float(np.mean([math.log2(fs[("adamw", M, sd)]["best_step"] / fs[("muon", M, sd)]["best_step"]) for sd in seeds])) for M in Ms]
fd = [float(np.mean([math.log2(fs[("adamw", M, sd)]["t10"] / fs[("muon", M, sd)]["t10"]) for sd in seeds])) for M in Ms]
print(f"- Muon reaches its best new-task error sooner by {2 ** min(pk):.1f}x to {2 ** max(pk):.1f}x across pool sizes "
      f"(geometric mean over seeds), and is 10% above its best sooner by {2 ** min(fd):.1f}x to {2 ** max(fd):.1f}x.")
