"""Run 8: the substitution-cipher task. Threshold M*, onset, and errors per setting.

usage: python scripts/run08_cipher.py results/run08_cipher reports/figs/run08
"""
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stats import threshold_log2M  # noqa: E402

LABEL = {"adamw": "AdamW 3e-4", "muon": "Muon 0.005", "muon_mlp": "Muon on MLP only", "muon_attn": "Muon on attention only"}
COL = {"adamw": "#2a78d6", "muon": "#eb6834", "muon_mlp": "#8e5a2b", "muon_attn": "#c0392b"}
SETTINGS = ("adamw", "muon_attn", "muon", "muon_mlp")


def load(d):
    runs = []
    for f in sorted(glob.glob(os.path.join(d, "*", "metrics.jsonl"))):
        rd = os.path.dirname(f)
        if not os.path.exists(os.path.join(rd, "done.json")):
            continue
        rows = [json.loads(l) for l in open(f)]
        runs.append(dict(cfg=json.load(open(os.path.join(rd, "config.json"))), base=json.load(open(os.path.join(rd, "baselines.json"))), rows=rows))
    return runs


def setting(c):
    return f"{c['opt']}_{c['route']}" if c["route"] != "all" else c["opt"]


def tail(r, k, n=5):
    return float(np.mean([row[k] for row in r["rows"][-n:] if k in row]))


def onset(r):
    """First step at which accuracy on repeated letters (new ciphers) exceeds 0.5: the in-context copy circuit."""
    for row in r["rows"]:
        if row.get("acc_ood_seen", 0) > 0.5:
            return row["step"]
    return None


def main():
    res, figs = sys.argv[1:3]
    os.makedirs(figs, exist_ok=True)
    runs = load(res)
    by = defaultdict(dict)
    for r in runs:
        by[setting(r["cfg"])][(r["cfg"]["seed"], r["cfg"]["M"])] = r
    Ms = sorted({k[1] for d in by.values() for k in d if k[1] > 0})
    out = ["## Cipher task: threshold M* at 15,000 steps\n", "| setting | runs | M* | log2 M* per seed | onset step (median, unlimited pool) |", "|---|---|---|---|---|"]
    thr = {}
    for s in SETTINGS:
        if s not in by:
            continue
        d = by[s]
        seeds = sorted({k[0] for k in d})
        t = []
        for sd in seeds:
            ms = [M for M in Ms if (sd, M) in d]
            if len(ms) >= 2:
                t.append(threshold_log2M(ms, [tail(d[(sd, M)], "alpha_ood") for M in ms]))
        thr[s] = t
        on = [onset(d[(sd, 0)]) for sd in seeds if (sd, 0) in d]
        on = [o for o in on if o is not None]
        out.append(f"| {LABEL[s]} | {len(d)} | {2 ** np.nanmean(t):.0f} | {' '.join(f'{v:.2f}' for v in t)} | {int(np.median(on)) if on else 'n/a'} |")
    if "adamw" in thr and "muon" in thr:
        n = min(len(thr["adamw"]), len(thr["muon"]))
        dd = np.array(thr["muon"][:n]) - np.array(thr["adamw"][:n])
        out.append(f"\nMuon minus AdamW: {np.nanmean(dd):+.2f} doublings ({' '.join(f'{v:+.2f}' for v in dd)})")
    if "muon_attn" in thr and "muon" in thr:
        n = min(len(thr["muon_attn"]), len(thr["muon"]))
        dd = np.array(thr["muon"][:n]) - np.array(thr["muon_attn"][:n])
        out.append(f"Full Muon minus Muon-on-attention (effect of Muon on the MLP): {np.nanmean(dd):+.2f} doublings ({' '.join(f'{v:+.2f}' for v in dd)})")
    if "muon_mlp" in thr and "adamw" in thr:
        n = min(len(thr["adamw"]), len(thr["muon_mlp"]))
        dd = np.array(thr["muon_mlp"][:n]) - np.array(thr["adamw"][:n])
        out.append(f"Muon-on-MLP minus AdamW: {np.nanmean(dd):+.2f} doublings ({' '.join(f'{v:+.2f}' for v in dd)})")

    out.append("\n## Per pool size (mean over seeds): alpha on new ciphers | unseen-letter CE on new ciphers minus optimal | unseen-letter accuracy on pool ciphers (memorisation; dMMSE-style optimum in brackets)\n")
    out.append("| M | " + " | ".join(LABEL[s] for s in SETTINGS if s in by) + " |")
    out.append("|---|" + "---|" * len([s for s in by]))
    for M in Ms + [0]:
        cells = []
        for s in SETTINGS:
            if s not in by:
                continue
            rs = [r for (sd, m), r in by[s].items() if m == M]
            if not rs:
                cells.append("-"); continue
            al = np.mean([tail(r, "alpha_ood") for r in rs]) if M > 0 else float("nan")
            ce = np.mean([tail(r, "ce_ood_unseen") - r["base"]["ce_icl_ood_unseen"] for r in rs])
            acc = np.mean([tail(r, "acc_id_unseen") for r in rs])
            opt = np.mean([r["base"]["acc_mem_id_unseen"] for r in rs])
            cells.append(f"{al:.2f} | {ce:+.2f} | {acc:.2f} ({opt:.2f})")
        out.append(f"| {'unlimited' if M == 0 else M} | " + " | ".join(cells) + " |")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for s in SETTINGS:
        if s not in by:
            continue
        for ax, key, lab in zip(axes, ("alpha_ood", "ce_ood_unseen", "acc_id_unseen"),
                                ("alpha on new ciphers (1 = memoriser)", "CE on unseen letters, new ciphers (dotted: optimal)", "accuracy on unseen letters, pool ciphers")):
            mu, se = [], []
            for M in Ms:
                v = [tail(r, key) for (sd, m), r in by[s].items() if m == M]
                mu.append(np.mean(v)); se.append(np.std(v) / max(1, np.sqrt(len(v))))
            ax.errorbar(Ms, mu, yerr=se, fmt="o-", color=COL[s], label=LABEL[s], capsize=2)
            ax.set_xscale("log", base=2); ax.set_xlabel("pool size M"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
    axes[0].axhline(0.5, color="k", ls=":")
    r0 = next(iter(next(iter(by.values())).values()))
    axes[1].axhline(r0["base"]["ce_icl_ood_unseen"], color="k", ls=":")
    axes[0].legend(fontsize=8)
    fig.suptitle("Substitution-cipher task, 15,000 steps (mean ± s.e. over seeds)")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "cipher_vs_pool_size.png"), dpi=130)

    # training curves at a few pool sizes
    show = [M for M in (64, 256, 1024) if M in Ms]
    fig, axes = plt.subplots(1, len(show), figsize=(5 * len(show), 4), squeeze=False)
    for ax, M in zip(axes[0], show):
        for s in SETTINGS:
            if s not in by:
                continue
            for i, ((sd, m), r) in enumerate(sorted(by[s].items())):
                if m != M:
                    continue
                st = [row["step"] for row in r["rows"]]
                ax.plot(np.maximum(st, 1), [row["ce_ood_unseen"] - r["base"]["ce_icl_ood_unseen"] for row in r["rows"]],
                        color=COL[s], lw=1, alpha=0.8, label=LABEL[s] if sd == 0 else None)
        ax.set_xscale("log"); ax.set_title(f"M = {M}"); ax.set_xlabel("step"); ax.set_ylabel("unseen-letter CE on new ciphers minus optimal"); ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "cipher_curves.png"), dpi=130)
    txt = "\n".join(out)
    print(txt)
    open(os.path.join(figs, "run08_table.md"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
