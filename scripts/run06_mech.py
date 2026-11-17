"""Run 6 (mechanism): which clock explains Muon's faster memorisation?

For every setting: threshold M* at the end (pools 16..256, 3 seeds), then M*(t) over training
plotted against four clocks: training step, cumulative weight-path length, distance from
initialisation ||W - W0||_F, and training loss. The clock on which the settings collapse is
the one that explains the gap.
usage: python scripts/run06_mech.py results/run06_mech reports/figs/run06
"""
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze import load, onset_step  # noqa: E402
from stats import tail_mean, threshold_log2M  # noqa: E402


def setting(c):
    o = c["opt"]
    if o == "muon":
        if c.get("route", "all") != "all":
            return f"muon_{c['route']}"
        if c.get("fro_match"):
            return f"muon_ns{c['ns_steps']}fm"
        return "muon_base"
    return {"adamw": "adamw_base", "adamw_fn": "adamw_fn"}.get(o, o)


ORDER = ["adamw_base", "adamw_fn", "muon_ns0fm", "muon_ns2fm", "muon_ns5fm", "muon_base", "muon_attn", "muon_mlp"]
LABEL = {"adamw_base": "AdamW 3e-4", "adamw_fn": "Adam direction, Muon step size", "muon_ns0fm": "Muon, 0 NS steps (no orthogonalisation)",
         "muon_ns2fm": "Muon, 2 NS steps", "muon_ns5fm": "Muon, 5 NS steps (norm-matched)", "muon_base": "Muon 0.005",
         "muon_attn": "Muon on attention only", "muon_mlp": "Muon on MLP only"}
COLS = dict(zip(ORDER, ["#2a78d6", "#7b3fd6", "#1baf7a", "#8ab800", "#e0a000", "#eb6834", "#c0392b", "#8e5a2b"]))


def dfro(row, which=None):
    v = [s[6] ** 2 for k, s in row["spec"].items() if which is None or (k.split(".")[1] in "qkvo") == (which == "attn")]
    return float(np.sqrt(np.sum(v)))


def main():
    res, figs = sys.argv[1], sys.argv[2]
    os.makedirs(figs, exist_ok=True)
    runs = [r for r in load(res) if r["done"] is not None and r["done"]["status"] == "ok"]
    by = defaultdict(dict)  # setting -> (seed, M) -> run
    for r in runs:
        by[setting(r["cfg"])][(r["cfg"]["seed"], r["cfg"]["M"])] = r
    out = []
    out.append("## Threshold M* at 30,000 steps (pools 16 to 256)\n")
    out.append("| setting | runs | M* (mean) | log2 M* per seed | onset step (median, M=256) | path length, attn / mlp (M=64) | ||W-W0||_F attn / mlp (M=64) |")
    out.append("|---|---|---|---|---|---|---|")
    thr = {}
    for s in ORDER:
        if s not in by:
            continue
        d = by[s]
        seeds = sorted({k[0] for k in d})
        Ms = sorted({k[1] for k in d})
        t = []
        for sd in seeds:
            ms = [M for M in Ms if (sd, M) in d]
            if len(ms) < 2:
                continue
            t.append(threshold_log2M(ms, [tail_mean(d[(sd, M)], "alpha_ood") for M in ms]))
        thr[s] = t
        on = [onset_step(d[(sd, 256)]) for sd in seeds if (sd, 256) in d]
        on = [o for o in on if o is not None]
        p64 = [d[(sd, 64)]["rows"][-1] for sd in seeds if (sd, 64) in d]
        pa = np.mean([r["path_attn"] for r in p64]) if p64 else np.nan
        pm = np.mean([r["path_mlp"] for r in p64]) if p64 else np.nan
        da = np.mean([dfro(r, "attn") for r in p64]) if p64 else np.nan
        dm = np.mean([dfro(r, "mlp") for r in p64]) if p64 else np.nan
        out.append(f"| {LABEL[s]} | {len(d)} | {2 ** np.nanmean(t):.0f} | {' '.join(f'{v:.2f}' for v in t)} | "
                   f"{int(np.median(on)) if on else 'n/a'} | {pa:.1f} / {pm:.1f} | {da:.2f} / {dm:.2f} |")

    # final errors table
    out.append("\n## New-task error / ridge and pool-task error / dMMSE at the end (mean over seeds)\n")
    Ms = sorted({k[1] for d in by.values() for k in d})
    out.append("| M | " + " | ".join(LABEL[s] for s in ORDER if s in by) + " |")
    out.append("|---|" + "---|" * len([s for s in ORDER if s in by]))
    for M in Ms:
        cells = []
        for s in ORDER:
            if s not in by:
                continue
            rs = [r for (sd, m), r in by[s].items() if m == M]
            vo = np.mean([tail_mean(r, "mse_ood_last") / r["base"]["ridge_ood_mse_pos"][-1] for r in rs]) if rs else np.nan
            vi = np.mean([tail_mean(r, "mse_id_last") / r["base"]["dmmse_id_mse_pos"][-1] for r in rs]) if rs else np.nan
            cells.append(f"{vo:.2f}x / {vi:.2f}x")
        out.append(f"| {M} | " + " | ".join(cells) + " |")

    # M*(t) against four clocks, pools 16..256 shared across seeds
    clocks = {"step": lambda row: max(row["step"], 1),
              "weight path length (attn + mlp)": lambda row: row["path_attn"] + row["path_mlp"],
              "distance from init ||W-W0||_F": lambda row: dfro(row),
              "training loss": lambda row: row["train_loss"] if row["train_loss"] is not None else np.nan}
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for ax, (cname, cf) in zip(axes, clocks.items()):
        for s in ORDER:
            if s not in by:
                continue
            d = by[s]
            seeds = sorted({k[0] for k in d})
            xs_all, ys_all = [], []
            for sd in seeds:
                ms = sorted(M for (q, M) in d if q == sd)
                if len(ms) < 2:
                    continue
                steps = [row["step"] for row in d[(sd, ms[0])]["rows"]]
                xs, ys = [], []
                for i, st in enumerate(steps):
                    if st < 300:
                        continue
                    try:
                        al = [d[(sd, M)]["rows"][i]["alpha_ood"] for M in ms]
                        x = np.mean([cf(d[(sd, M)]["rows"][i]) for M in ms])
                    except (IndexError, KeyError):
                        continue
                    y = threshold_log2M(ms, al)
                    if np.isfinite(y) and np.isfinite(x):
                        xs.append(x)
                        ys.append(y)
                xs_all.append(np.array(xs))
                ys_all.append(np.array(ys))
            if not xs_all:
                continue
            # average over seeds on a common grid in log x
            lo = max(a.min() for a in xs_all if len(a))
            hi = min(a.max() for a in xs_all if len(a))
            if not (hi > lo > 0):
                continue
            grid = np.geomspace(lo, hi, 60)
            Y = np.array([np.interp(grid, a if cname != "training loss" else a[::-1], b if cname != "training loss" else b[::-1]) for a, b in zip(xs_all, ys_all)])
            ax.plot(grid, Y.mean(0), color=COLS[s], lw=1.6, label=LABEL[s])
        ax.set_xscale("log")
        ax.set_xlabel(cname)
        ax.set_ylabel("log2 M* (pool size where alpha = 0.5)")
        ax.grid(alpha=0.3)
    axes[-1].invert_xaxis()
    axes[0].legend(fontsize=7)
    fig.suptitle("Threshold over training against four clocks (mean of seeds). The clock that collapses the curves explains the gap.")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "threshold_vs_clocks.png"), dpi=130)

    # new-task error over training at M=128, all settings
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, M in zip(axes, (64, 128, 256)):
        for s in ORDER:
            if s not in by:
                continue
            rs = [r for (sd, m), r in by[s].items() if m == M]
            for i, r in enumerate(rs):
                st = np.array([row["step"] for row in r["rows"]])
                y = np.array([row["mse_ood_last"] for row in r["rows"]]) / r["base"]["ridge_ood_mse_pos"][-1]
                ax.plot(np.maximum(st, 1), y, color=COLS[s], lw=1, alpha=0.7, label=LABEL[s] if i == 0 else None)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"M = {M}"); ax.set_xlabel("step"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("new-task error / ridge"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "new_task_error.png"), dpi=130)

    # alpha vs M at the end
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for s in ORDER:
        if s not in by:
            continue
        mu = [np.mean([tail_mean(r, "alpha_ood") for (sd, m), r in by[s].items() if m == M]) for M in Ms]
        ax.plot(Ms, mu, "o-", color=COLS[s], label=LABEL[s])
    ax.axhline(0.5, color="k", ls=":"); ax.set_xscale("log", base=2); ax.set_xlabel("pool size M"); ax.set_ylabel("alpha on new tasks (1 = memoriser)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(os.path.join(figs, "alpha_vs_M.png"), dpi=130)

    # paired differences vs adamw_base and vs muon_base
    out.append("\n## Paired threshold differences (doublings of M*, per seed)\n")
    out.append("| setting | minus AdamW 3e-4 | minus Muon 0.005 |")
    out.append("|---|---|---|")
    for s in ORDER:
        if s in thr and "adamw_base" in thr and "muon_base" in thr:
            n = min(len(thr[s]), len(thr["adamw_base"]), len(thr["muon_base"]))
            da = np.array(thr[s][:n]) - np.array(thr["adamw_base"][:n])
            dm = np.array(thr[s][:n]) - np.array(thr["muon_base"][:n])
            out.append(f"| {LABEL[s]} | {np.nanmean(da):+.2f} ({' '.join(f'{v:+.2f}' for v in da)}) | {np.nanmean(dm):+.2f} ({' '.join(f'{v:+.2f}' for v in dm)}) |")
    txt = "\n".join(out)
    print(txt)
    open(os.path.join(figs, "run06_table.md"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
