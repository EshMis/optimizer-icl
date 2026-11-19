"""Tables and a figure from scripts/measure_cost.py output.

usage: python scripts/cost_report.py reports/figs/cost
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {"sgdm": "SGD with momentum", "adamw": "AdamW", "adamw_fn": "Adam direction, Muon step size*",
         "signum": "Signum", "muon_ns0fm": "Muon, 0 Newton-Schulz steps", "muon_ns2fm": "Muon, 2 Newton-Schulz steps",
         "muon": "Muon (5 Newton-Schulz steps)"}
ORDER = ["sgdm", "adamw", "signum", "muon_ns0fm", "muon_ns2fm", "muon", "adamw_fn"]
# fixed categorical slots, validated with the dataviz palette validator (light surface)
CHART = [("adamw", "AdamW", "#2a78d6"), ("muon", "Muon", "#eb6834"), ("signum", "Signum", "#1baf7a"), ("sgdm", "SGD", "#eda100")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e5e0"


def main():
    d = sys.argv[1]
    meta = json.load(open(os.path.join(d, "cost.json")))
    R = {(r["width"], r["setting"]): r for r in meta["results"]}
    Ws = sorted({r["width"] for r in meta["results"]})
    out = [f"Measured on {meta['device']}, PyTorch {meta['torch']}, batch 256 prompts of 32 tokens, "
           f"32-bit floats (Newton-Schulz in bfloat16), median of {meta['steps']} timed steps.\n"]

    out.append("## Time per training step (milliseconds)\n")
    out.append("Model = forward + backward. Optimizer = the step for the attention and MLP matrices. The AdamW "
               "step for all other parameters is the same in every setting and is listed once per width.\n")
    out.append("| Optimizer | " + " | ".join(f"width {W}: model / optimizer / overhead" for W in Ws) + " |")
    out.append("|---|" + "---|" * len(Ws))
    for k in ORDER:
        cells = []
        for W in Ws:
            r = R[(W, k)]
            cells.append(f"{r['ms_model']:.1f} / {r['ms_opt']:.2f} / {100 * r['ms_opt'] / r['ms_model']:.1f}%")
        out.append(f"| {LABEL[k]} | " + " | ".join(cells) + " |")
    out.append("| AdamW for the other parameters | " + " | ".join(f"{R[(W, 'adamw')]['ms_aux']:.2f}" for W in Ws) + " |")

    out.append("\n## Matrix-product FLOPs per training step (billions)\n")
    out.append("Counted by PyTorch's FLOP counter, which sees matrix products only; element-wise optimizer work "
               "(AdamW, Signum, SGD) reads as zero here and appears in the times above.\n")
    out.append("| Optimizer | " + " | ".join(f"width {W}: model / optimizer / overhead" for W in Ws) + " |")
    out.append("|---|" + "---|" * len(Ws))
    for k in ORDER:
        cells = []
        for W in Ws:
            r = R[(W, k)]
            cells.append(f"{r['flops_model'] / 1e9:.1f} / {r['flops_opt'] / 1e9:.2f} / {100 * r['flops_opt'] / r['flops_model']:.1f}%")
        out.append(f"| {LABEL[k]} | " + " | ".join(cells) + " |")

    out.append("\n## Optimizer state memory for the attention and MLP matrices\n")
    out.append("| Optimizer | " + " | ".join(f"width {W} (MB)" for W in Ws) + " | bytes per weight |")
    out.append("|---|" + "---|" * (len(Ws) + 1))
    for k in ORDER:
        bpw = R[(Ws[0], k)]["state_bytes"] / R[(Ws[0], k)]["n_block"]
        out.append(f"| {LABEL[k]} | " + " | ".join(f"{R[(W, k)]['state_bytes'] / 1e6:.1f}" for W in Ws) + f" | {bpw:.0f} |")
    out.append("\n*The Adam-direction control is a per-matrix Python loop written for the ablation, not a fused "
               "implementation, so its time reflects that code rather than the method.")
    txt = "\n".join(out)
    print(txt)
    open(os.path.join(d, "cost_table.md"), "w").write(txt + "\n")

    # figure: optimizer time as a share of the model's forward + backward time, against width
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    for k, lab, col in CHART:
        y = [100 * R[(W, k)]["ms_opt"] / R[(W, k)]["ms_model"] for W in Ws]
        ax.plot(Ws, y, color=col, lw=2, marker="o", ms=8, mec="#fcfcfb", mew=2,
                label=f"{lab}: {y[-1]:.0f}% at width {Ws[-1]}", zorder=3)
    ax.set_xscale("log", base=2); ax.set_xticks(Ws); ax.set_xticklabels([str(W) for W in Ws])
    ax.set_xlim(Ws[0] / 1.3, Ws[-1] * 1.3)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("model width", color=MUTED); ax.set_ylabel("optimizer time, % of forward + backward", color=MUTED)
    ax.set_title("Extra time each optimizer adds per training step", loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5), labelcolor=INK)
    fig.text(0.01, 0.01, f"{meta['device']}, median of {meta['steps']} steps; block matrices only. At width 64 the "
             "fixed cost of launching each GPU operation dominates every step.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(d, "optimizer_overhead.png"), dpi=150, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
