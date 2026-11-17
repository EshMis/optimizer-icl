"""Compute accounting: the task-diversity threshold M*(t) against training FLOPs, across widths.

FLOPs per step = 3 x forward (forward + backward), forward = 2 x (matrix parameters) x tokens
+ attention score and mixing terms. Muon adds 5 quintic Newton-Schulz iterations per block
matrix per step. M*(t) is computed at every evaluation from the pools of each seed and
averaged over seeds, and kept from the in-context-learning onset of the largest pool (latest
seed), then from the curve's low point; before that, alpha is near 0.5 at every pool size and
the threshold is noise. Width 128 comes from Run 6, the other widths from Run 7a.
usage: python scripts/flops.py results/run06_mech results/run07a_width reports/figs/flops
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
from icl.model import ICLTransformer  # noqa: E402
from scripts.run06_mech import setting  # noqa: E402
from stats import threshold_log2M  # noqa: E402

COL = {"adamw": "#2a78d6", "muon": "#eb6834"}
WID_LS = {64: ":", 128: "-.", 256: "--", 512: "-"}


def ns_flops(m, n):
    m, n = min(m, n), max(m, n)
    return 2 * m * m * n + 2 * m ** 3 + 2 * m * m * n     # X X^T, A A, B X


def flops_per_step(c, opt):
    mdl = ICLTransformer(c["d"], c["K"], c["n_layer"], c["n_embd"], c["n_head"])
    H = mdl.hidden_matrices()
    T, B, W, L = 2 * c["K"], c["batch"], c["n_embd"], c["n_layer"]
    nmat = sum(p.numel() for p in H.values()) + sum(
        p.numel() for n, p in mdl.named_parameters() if n.startswith(("read_in", "read_out")) and p.dim() == 2)
    fwd = 2 * nmat * B * T + L * 4 * T * T * W * B
    extra = 5 * sum(ns_flops(*p.shape) for p in H.values()) if opt == "muon" else 0
    return 3 * fwd + extra, extra


def main():
    r6, r7, figs = sys.argv[1:4]
    os.makedirs(figs, exist_ok=True)
    ok = lambda rs: [r for r in rs if r["done"] is not None and r["done"]["status"] == "ok"]
    runs = ok(load(r7)) + [r for r in ok(load(r6)) if setting(r["cfg"]) in ("adamw_base", "muon_base")]
    by = defaultdict(lambda: defaultdict(dict))     # (opt, W) -> seed -> M -> run
    for r in runs:
        c = r["cfg"]
        if c["M"] > 0:
            by[(c["opt"], c["n_embd"])][c["seed"]][c["M"]] = r
    curves, fps = {}, {}
    out = ["## Training compute per step\n", "| width | optimizer | GFLOPs per step | Newton-Schulz share |", "|---|---|---|---|"]
    for (o, W), seeds in sorted(by.items(), key=lambda k: (k[0][1], k[0][0])):
        cfg = next(iter(next(iter(seeds.values())).values()))["cfg"]
        f, extra = flops_per_step(cfg, o)
        fps[(o, W)] = f
        out.append(f"| {W} | {o} | {f / 1e9:.1f} | {100 * extra / f:.1f}% |")
        per_seed, onsets = [], []
        for sd, byM in seeds.items():
            on = onset_step(byM[max(byM)])
            onsets.append(on if on is not None else 10 ** 9)
            Ms = sorted(byM)
            if len(Ms) < 2:
                continue
            n = min(len(byM[M]["rows"]) for M in Ms)
            steps = np.array([byM[Ms[0]]["rows"][i]["step"] for i in range(n)])
            y = np.array([threshold_log2M(Ms, [byM[M]["rows"][i]["alpha_ood"] for M in Ms]) for i in range(n)])
            per_seed.append((steps, y))
        n = min(len(s) for s, _ in per_seed)
        steps = per_seed[0][0][:n]
        Y = np.array([y[:n] for _, y in per_seed])
        mu = np.nanmean(Y, 0)
        # Before in-context learning forms, alpha sits near 0.5 at every pool size and the crossing
        # is noise. Start after the largest pool's ICL onset (latest seed), then from the curve's low point.
        keep = (steps >= max(onsets)) & np.isfinite(mu)
        st, mu = steps[keep], mu[keep]
        i0 = int(np.argmin(mu)) if len(mu) else 0
        curves[(o, W)] = (st[i0:], mu[i0:])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for (o, W), (st, mu) in sorted(curves.items()):
        lab = f"{o}, width {W}"
        axes[0].plot(st, 2 ** mu, color=COL[o], ls=WID_LS[W], label=lab)
        axes[1].plot(st * fps[(o, W)], 2 ** mu, color=COL[o], ls=WID_LS[W], label=lab)
    for ax, xl in zip(axes, ("training step", "training compute (FLOPs)")):
        ax.set_xscale("log"); ax.set_yscale("log", base=2); ax.set_xlabel(xl); ax.grid(alpha=0.3)
        ax.set_ylabel("threshold M* (tasks)")
    axes[0].set_title("Threshold over training, by step")
    axes[1].set_title("Same curves against compute")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "threshold_vs_flops.png"), dpi=130)

    # compute needed to reach given thresholds, and Muon's compute advantage at each width
    out.append("\n## Compute to reach a threshold (first evaluation where the seed-mean M* reaches it)\n")
    targets = (32, 48, 64)
    out.append("| width | " + " | ".join(f"AdamW to M*={t} | Muon to M*={t} | AdamW / Muon" for t in targets) + " |")
    out.append("|---|" + "---|" * (3 * len(targets)))
    Ws = sorted({W for _, W in curves})
    for W in Ws:
        cells = []
        for t in targets:
            v = {}
            for o in ("adamw", "muon"):
                st, mu = curves.get((o, W), (np.array([]), np.array([])))
                hit = np.nonzero(2 ** mu >= t)[0]
                v[o] = st[hit[0]] * fps[(o, W)] if len(hit) else None
            fa, fm = v["adamw"], v["muon"]
            cells += [f"{fa:.2e}" if fa else "not reached", f"{fm:.2e}" if fm else "not reached",
                      f"{fa / fm:.1f}x" if fa and fm else "-"]
        out.append(f"| {W} | " + " | ".join(cells) + " |")
    txt = "\n".join(out)
    print(txt)
    open(os.path.join(figs, "flops_table.md"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
