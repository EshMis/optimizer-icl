"""Run 7: (a) threshold M* against model width for AdamW and Muon; (b) Signum at Muon's norm.

Width 128 comes from Run 6 (adamw_base, muon_base); the other widths from Run 7a.
usage: python scripts/run07_capacity.py results/run06_mech results/run07a_width results/run07b_signum reports/figs/run07
"""
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze import load  # noqa: E402
from stats import tail_mean, threshold_log2M  # noqa: E402
from scripts.run06_mech import setting, LABEL  # noqa: E402

COL = {"adamw": "#2a78d6", "muon": "#eb6834"}


def thresholds(runs, key):
    """key(run) -> group; returns group -> list of log2 M* per seed."""
    by = defaultdict(dict)
    for r in runs:
        by[key(r)][(r["cfg"]["seed"], r["cfg"]["M"])] = r
    out = {}
    for g, d in by.items():
        seeds = sorted({k[0] for k in d})
        t = []
        for sd in seeds:
            ms = sorted(M for (q, M) in d if q == sd and M > 0)
            if len(ms) >= 2:
                t.append(threshold_log2M(ms, [tail_mean(d[(sd, M)], "alpha_ood") for M in ms]))
        out[g] = t
    return out


def main():
    r6, r7a, r7b, figs = sys.argv[1:5]
    os.makedirs(figs, exist_ok=True)
    ok = lambda rs: [r for r in rs if r["done"] is not None and r["done"]["status"] == "ok"]
    base = [r for r in ok(load(r6)) if setting(r["cfg"]) in ("adamw_base", "muon_base")]
    width = ok(load(r7a)) + base
    thr = thresholds(width, lambda r: (r["cfg"]["opt"], r["cfg"]["n_embd"]))
    out = ["## Threshold M* against width (30,000 steps, pools 16 to 256)\n",
           "| width | params | AdamW M* | per seed (log2) | Muon M* | per seed (log2) | gap (doublings) |", "|---|---|---|---|---|---|---|"]
    Ws = sorted({k[1] for k in thr})
    curves = {"adamw": [], "muon": []}
    for W in Ws:
        a, m = thr.get(("adamw", W), []), thr.get(("muon", W), [])
        n = min(len(a), len(m))
        gap = np.nanmean(np.array(m[:n]) - np.array(a[:n])) if n else np.nan
        npar = next((r["cfg"]["n_params"] for r in width if r["cfg"]["n_embd"] == W), 0)
        out.append(f"| {W} | {npar / 1e6:.2f}M | {2 ** np.nanmean(a):.0f} | {' '.join(f'{v:.2f}' for v in a)} | "
                   f"{2 ** np.nanmean(m):.0f} | {' '.join(f'{v:.2f}' for v in m)} | {gap:+.2f} |")
        curves["adamw"].append((W, np.nanmean(a), np.nanstd(a)))
        curves["muon"].append((W, np.nanmean(m), np.nanstd(m)))
    # slopes in log-log
    for o, c in curves.items():
        c = [x for x in c if np.isfinite(x[1])]
        if len(c) >= 2:
            s = np.polyfit(np.log2([x[0] for x in c]), [x[1] for x in c], 1)[0]
            out.append(f"\n{o}: M* grows as width^{s:.2f} (slope of log2 M* against log2 width)")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for o, c in curves.items():
        c = [x for x in c if np.isfinite(x[1])]
        ax.errorbar([x[0] for x in c], [2 ** x[1] for x in c], yerr=[2 ** x[1] * np.log(2) * x[2] for x in c],
                    fmt="o-", color=COL[o], label=o, capsize=3)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("model width (n_embd)"); ax.set_ylabel("threshold M* (tasks)")
    ax.set_title("Task-diversity threshold against width, 30k steps"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "threshold_vs_width.png"), dpi=130)

    # Signum reconciliation
    sig = ok(load(r7b))
    r6all = ok(load(r6))
    grp = thresholds(r6all + sig, lambda r: "signum_fm" if r["cfg"]["opt"] == "signum" else setting(r["cfg"]))
    out.append("\n## Signum at Muon's Frobenius norm, next to Run 6 (30,000 steps)\n")
    out.append("| setting | M* | log2 M* per seed |"); out.append("|---|---|---|")
    for g, lab in (("adamw_base", "AdamW 3e-4"), ("muon_ns0fm", "Muon, 0 NS steps (normalised momentum)"),
                   ("signum_fm", "Signum, norm-matched"), ("muon_base", "Muon 0.005")):
        t = grp.get(g, [])
        if t:
            out.append(f"| {lab} | {2 ** np.nanmean(t):.0f} | {' '.join(f'{v:.2f}' for v in t)} |")
    txt = "\n".join(out)
    print(txt)
    open(os.path.join(figs, "run07_table.md"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
