"""The figures used in WRITEUP.md, drawn from the saved metrics.

usage: python scripts/writeup_figs.py results/run02_main results/run03_long reports/figs/writeup

Conventions shared with stats.py: a run's end value is the mean of its last 5 evaluations; the
threshold M* is where alpha on new tasks crosses 0.5, interpolated in log2 M; error bars are the
standard error over seeds.
"""
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze import COL, load, series  # noqa: E402
from stats import tail_mean, threshold_log2M  # noqa: E402
from scripts.examples_needed import at, measures  # noqa: E402

INK, MUTED = "#0b0b0b", "#52514e"
NAME = {"adamw": "AdamW", "muon": "Muon", "signum": "Signum", "sgdm": "SGD"}
SEEDS = range(6)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
                     "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white"})


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-6)


def pick(runs, opt, lr):
    return {(r["cfg"]["M"], r["cfg"]["seed"]): r for r in runs
            if r["cfg"]["opt"] == opt and close(r["cfg"]["lr"], lr) and r["cfg"]["wd"] == 0}


def mean_se(v):
    v = np.asarray(v, float)
    return v.mean(), v.std() / math.sqrt(len(v))


def thr_per_seed(cell, Ms):
    return np.array([threshold_log2M(Ms, [tail_mean(cell[(M, s)], "alpha_ood") for M in Ms]) for s in SEEDS])


def fig_pool_size(runs, out):
    Ms = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    sets = [("adamw", 3e-4), ("muon", 0.005)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    for o, lr in sets:
        cell = pick(runs, o, lr)
        t = 2 ** np.mean(thr_per_seed(cell, Ms))
        for ax, f in zip(axes, (lambda r: tail_mean(r, "alpha_ood"),
                                lambda r: tail_mean(r, "mse_ood_last") / r["base"]["ridge_ood_mse_pos"][-1])):
            ms = [mean_se([f(cell[(M, s)]) for s in SEEDS]) for M in Ms]
            ax.errorbar(Ms, [m[0] for m in ms], yerr=[m[1] for m in ms], color=COL[o], lw=2, marker="o", ms=5,
                        capsize=2, label=f"{NAME[o]} (learning rate {lr:g})")
        axes[0].axvline(t, color=COL[o], lw=1, ls="--")
        left = o == "adamw"
        axes[0].text(t / 1.08 if left else t * 1.08, 0.2 if left else 0.66,
                     f"{NAME[o]} switches\nat about {t:.0f} tasks", fontsize=8, color=INK, va="top",
                     ha="right" if left else "left")
    axes[0].axhline(0.5, color=MUTED, lw=0.8, ls=":")
    axes[0].set_ylabel("alpha on new tasks\n(higher = behaves like a memoriser)")
    axes[0].set_title("Which strategy the model ends up using", fontsize=10, loc="left")
    axes[1].axhline(1, color=MUTED, lw=0.8, ls=":")
    axes[1].text(16, 1.06, "optimal (ridge regression)", fontsize=8, color=MUTED)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("error on new tasks / optimal error")
    axes[1].set_title("How well it does on tasks it never saw", fontsize=10, loc="left")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ms)
        ax.set_xticklabels([f"{m:,}" for m in Ms], fontsize=7.5)
        ax.set_xlabel("number of distinct training tasks (pool size)")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("After 30,000 training steps (mean of 6 seeds, bars: standard error)", fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_pool_size.png"), dpi=150)
    plt.close(fig)


def fig_clock(runs3, out, shift=4.0):
    by = defaultdict(lambda: defaultdict(dict))
    for r in runs3:
        c = r["cfg"]
        by[c["opt"]][c["seed"]][c["M"]] = {row["step"]: row["alpha_ood"] for row in r["rows"]}
    seeds = sorted(set(by["adamw"]) & set(by["muon"]))
    Ms = sorted(by["adamw"][seeds[0]])
    steps = np.array(sorted(by["adamw"][seeds[0]][Ms[0]]))
    keep = steps >= 1000
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    mean = {}
    for o in ("adamw", "muon"):
        cur = np.array([[threshold_log2M(Ms, [by[o][sd][M][s] for M in Ms]) for s in steps] for sd in seeds])
        with np.errstate(all="ignore"):
            mean[o] = np.where(np.isfinite(cur).sum(0) == len(seeds), np.nanmean(cur, axis=0), np.nan)
        ax.plot(steps[keep], mean[o][keep], color=COL[o], lw=2, label=NAME[o])
    ax.plot(steps[keep] / shift, mean["adamw"][keep], color=COL["adamw"], lw=1.6, ls=":",
            label=f"AdamW, moved {shift:g}x earlier")
    ax.set_xscale("log")
    ax.set_xlim(1500, 3.3e5)
    ticks = [5, 6, 7, 8]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{2 ** t}" for t in ticks])
    ax.set_xlabel("training step")
    ax.set_ylabel("threshold M* (tasks)\nbelow it the model memorises")
    ax.set_title("The threshold keeps rising, and Muon is about 4x ahead", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(0.99, 0.03, f"long runs, 300,000 steps, mean of {len(seeds)} seeds; smallest pool in this run is 32 tasks",
            transform=ax.transAxes, fontsize=7, color=MUTED, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_threshold_over_time.png"), dpi=150)
    plt.close(fig)


def fig_fade(runs3, out):
    show = [256, 1024]
    fig, axes = plt.subplots(1, len(show), figsize=(9.6, 3.7), sharey=True)
    for ax, M in zip(axes, show):
        for o in ("adamw", "muon"):
            rs = [r for r in runs3 if r["cfg"]["opt"] == o and r["cfg"]["M"] == M]
            ys = []
            for r in rs:
                s, y = series(r, "mse_ood_last")
                y = y / r["base"]["ridge_ood_mse_pos"][-1]
                ys.append(y)
                ax.plot(np.maximum(s, 1), y, color=COL[o], lw=0.7, alpha=0.3)
            mu = np.exp(np.mean(np.log(ys), axis=0))
            ax.plot(np.maximum(s, 1), mu, color=COL[o], lw=2, label=NAME[o])
            ok = s >= 300
            i = int(np.argmin(np.where(ok, mu, np.inf)))
            ax.plot([s[i]], [mu[i]], marker="v", ms=9, color=COL[o], mec="white", mew=1.5, zorder=5)
        ax.axhline(1, color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(100, 3.3e5)
        ax.set_ylim(0.8, 20)
        ax.set_xlabel("training step")
        ax.set_title(f"{M:,} training tasks", fontsize=10, loc="left")
    axes[0].set_ylabel("error on new tasks / optimal error")
    axes[0].set_yticks([1, 2, 5, 10])
    axes[0].set_yticklabels(["1", "2", "5", "10"])
    axes[1].plot([], [], marker="v", ms=8, color=MUTED, ls="", label="lowest point of the mean curve")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("In-context learning appears, peaks, then fades as the model memorises (thin lines: seeds)",
                 fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_rise_and_fade.png"), dpi=150)
    plt.close(fig)


def fig_ladder(runs, out):
    Ms = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    rows = [("sgdm", 0.1, "SGD with momentum\nplain gradient steps"),
            ("adamw", 1e-4, "AdamW, learning rate 1e-4\nper-entry rescaling"),
            ("adamw", 3e-4, "AdamW, learning rate 3e-4"),
            ("signum", 1e-4, "Signum, learning rate 1e-4\nsame step size for every entry"),
            ("signum", 3e-4, "Signum, learning rate 3e-4"),
            ("muon", 0.005, "Muon, learning rate 0.005\nsame step size in every direction"),
            ("muon", 0.01, "Muon, learning rate 0.01")]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    for i, (o, lr, lab) in enumerate(rows):
        y = len(rows) - 1 - i
        t = thr_per_seed(pick(runs, o, lr), Ms)
        ax.scatter(t, [y] * len(t), s=16, color=COL[o], alpha=0.4, zorder=2)
        ax.scatter([t.mean()], [y], s=90, color=COL[o], edgecolor="white", linewidth=2, zorder=3)
        ax.text(t.max() + 0.07, y, f"{2 ** t.mean():.0f} tasks", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[2] for r in rows][::-1], fontsize=8, color=INK)
    ticks = [24, 32, 48, 64, 96]
    ax.set_xticks([math.log2(t) for t in ticks])
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(4.4, 6.9)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("threshold M* after 30,000 steps (tasks; small dots: 6 seeds, large dot: mean)")
    ax.set_title("Each ingredient of Muon's update raises the threshold", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_what_in_muon.png"), dpi=150)
    plt.close(fig)


def fig_examples(runs, out):
    Ms = [32, 64, 128, 256, 512, 1024, 2048, 4096, 0]
    xs = list(range(len(Ms)))
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    for o, lr in (("adamw", 3e-4), ("muon", 0.005)):
        cell = pick(runs, o, lr)
        ms = [mean_se([measures(at(cell[(M, s)], slice(-5, None)), cell[(M, s)]["base"]["ridge_ood_mse_pos"])[2]
                       for s in SEEDS]) for M in Ms]
        ax.errorbar(xs, [m[0] for m in ms], yerr=[m[1] for m in ms], color=COL[o], lw=2, marker="o", ms=5, capsize=2,
                    label=f"{NAME[o]} (learning rate {lr:g})")
    ax.axhline(15, color=MUTED, lw=0.8, ls=":")
    ax.text(0, 15.25, "every example fully used", fontsize=8, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{m:,}" if m else "new task\nevery prompt" for m in Ms], fontsize=7.5)
    ax.set_ylim(0, 16.5)
    ax.set_xlabel("number of distinct training tasks (pool size)")
    ax.set_ylabel("what 15 examples are worth\n(examples optimal regression would need)")
    ax.set_title("With enough tasks, both use their examples equally well", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_examples.png"), dpi=150)
    plt.close(fig)


def main():
    d2, d3, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    ok = lambda r: r["done"] is not None and r["done"]["status"] == "ok"  # noqa: E731
    runs = [r for r in load(d2) if ok(r)]
    runs3 = [r for r in load(d3) if ok(r)]
    fig_pool_size(runs, out)
    fig_ladder(runs, out)
    fig_examples(runs, out)
    fig_clock(runs3, out)
    fig_fade(runs3, out)
    print("wrote", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
