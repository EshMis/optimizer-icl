"""Summarise one sweep: a markdown table plus standard figures.

usage: python analyze.py results/run01_calibrate --figs reports/figs/run01
Columns use errors at the last context position (15 examples seen), the proposal's headline.
"rel" = model error / ridge error on the same prompts (1.0 = as good as optimal regression).
"""
import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = {"adamw": "#1f77b4", "muon": "#d62728", "signum": "#2ca02c", "sgdm": "#9467bd"}


def load(d):
    runs = []
    for f in sorted(glob.glob(os.path.join(d, "*", "metrics.jsonl"))):
        rd = os.path.dirname(f)
        rows = [json.loads(line) for line in open(f)]
        if not rows:
            continue
        r = dict(id=os.path.basename(rd), cfg=json.load(open(os.path.join(rd, "config.json"))),
                 base=json.load(open(os.path.join(rd, "baselines.json"))), rows=rows,
                 done=json.load(open(os.path.join(rd, "done.json"))) if os.path.exists(os.path.join(rd, "done.json")) else None)
        runs.append(r)
    return runs


def series(r, key):
    return np.array([row["step"] for row in r["rows"]]), np.array([row[key] for row in r["rows"]], dtype=float)


def onset_step(r):
    """First step where new-task error at the last position is halfway from its start to ridge."""
    s, y = series(r, "mse_ood_last")
    ridge = r["base"]["ridge_ood_mse_pos"][-1]
    thr = (y[0] + ridge) / 2
    hit = np.nonzero(y <= thr)[0]
    return int(s[hit[0]]) if len(hit) else None


def fade(r):
    """Best new-task error over training, the step it happened, and final / best (1.0 = no fading)."""
    s, y = series(r, "mse_ood_last")
    i = int(np.argmin(y))
    return y[i], int(s[i]), float(np.mean(y[-5:]) / y[i])


def label(c):
    m = "inf" if c["M"] == 0 else f"2^{int(math.log2(c['M']))}" if c["M"] & (c["M"] - 1) == 0 else str(c["M"])
    return f"{c['opt']} lr={c['lr']:g}" + ("" if c["aux_lr"] == c["lr"] else f" aux={c['aux_lr']:g}") + f" M={m} seed={c['seed']}"


def table(runs):
    hdr = ("| run | status | min | new-task err (rel) | pool-task err (rel to dMMSE) | alpha new | alpha pool "
           "| ICL onset step | best new-task err @ step | fade (final/best) |")
    out = [hdr, "|" + "---|" * 10]
    for r in sorted(runs, key=lambda r: (r["cfg"]["M"] or 1 << 40, r["cfg"]["opt"], r["cfg"]["lr"])):
        last, b = r["rows"][-1], r["base"]
        st = "running" if r["done"] is None else r["done"]["status"]
        mins = f"{last['elapsed'] / 60:.0f}"
        rel_ood = last["mse_ood_last"] / b["ridge_ood_mse_pos"][-1]
        rel_id = last["mse_id_last"] / b["dmmse_id_mse_pos"][-1]
        on = onset_step(r)
        out.append(f"| {label(r['cfg'])} | {st} @{last['step']} | {mins} | {last['mse_ood_last']:.3f} ({rel_ood:.2f}x) | "
                   f"{last['mse_id_last']:.3f} ({rel_id:.2f}x) | {last['alpha_ood']:.2f} | {last['alpha_id']:.2f} | {on if on is not None else 'never'} | "
                   f"{fade(r)[0]:.3f} @ {fade(r)[1]} | {fade(r)[2]:.2f} |")
    return "\n".join(out)


def fig_curves(runs, path, key="mse_ood_last", ref="ridge_ood_mse_pos", title="", logy=True):
    Ms = sorted({r["cfg"]["M"] for r in runs}, key=lambda m: m or 1 << 40)
    fig, axes = plt.subplots(1, len(Ms), figsize=(4 * len(Ms), 3.2), squeeze=False)
    for ax, M in zip(axes[0], Ms):
        rs = [r for r in runs if r["cfg"]["M"] == M]
        lrs = sorted({r["cfg"]["lr"] for r in rs})
        for r in rs:
            s, y = series(r, key)
            o = r["cfg"]["opt"]
            same = sorted(x["cfg"]["lr"] for x in rs if x["cfg"]["opt"] == o)
            a = 0.35 + 0.65 * (same.index(r["cfg"]["lr"]) + 1) / len(same)
            ax.plot(np.maximum(s, 1), y, color=COL.get(o, "k"), alpha=a, lw=1.2, label=label(r["cfg"]).split(" M=")[0])
        if ref:
            ax.axhline(rs[0]["base"][ref][-1], color="k", ls="--", lw=0.8, label="ridge")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"M = {'inf' if M == 0 else M}")
        ax.set_xlabel("step")
        del lrs
    axes[0][0].set_ylabel(title or key)
    axes[0][-1].legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--figs", required=True)
    a = ap.parse_args()
    os.makedirs(a.figs, exist_ok=True)
    runs = load(a.results)
    print(table(runs))
    fig_curves(runs, os.path.join(a.figs, "new_task_error.png"), title="error on NEW tasks, last position")
    fig_curves(runs, os.path.join(a.figs, "pool_task_error.png"), key="mse_id_last", ref="dmmse_id_mse_pos",
               title="error on POOL tasks (dashed = dMMSE)")
    fig_curves(runs, os.path.join(a.figs, "alpha_new.png"), key="alpha_ood", ref=None, logy=False,
               title="alpha on new tasks (1 = memorising, 0 = ICL)")


if __name__ == "__main__":
    main()
