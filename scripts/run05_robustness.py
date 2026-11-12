"""Run 5 (robustness): apply the decision rules of PLAN.md "Run 5 design" exactly as written.

Reads the learning-rate grid from one results directory (Run 2's, where 5a and 5c wrote) and
the matched-weight-decay runs from another, and writes one markdown report plus two figures.

usage: python scripts/run05_robustness.py results/run02_main results/run05b_wd \
           reports/run05_robustness.md reports/figs/run05

Conventions shared with stats.py: a run's end value is the mean of its last 5 evaluations; the
threshold M* is where alpha on new tasks crosses 0.5, interpolated in log2 M; pairs are runs
with the same seed (same task pool, starting weights and batches).

Reading of the trainability filter (fixed before any Run 5 result was looked at): a setting
counts as trainable when EVERY infinite-pool run ends within 1.3x of ridge. The seed mean is
printed next to it so a reader can see whether the other reading would change anything.

Added after looking at the results, and labelled that way in the report: the per-pool-size
stability check (does a run end within 1.3x of ridge on prompts from its own pool?), the
re-run of rule 1 over settings that pass it, the direct comparison of Muon's top learning rate
with AdamW's at the pool sizes where both train, and the onset-matched comparison.
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze import COL, fade, load, onset_step  # noqa: E402
from stats import boot_ci, sign_flip_p, tail_mean, threshold_log2M  # noqa: E402
from scripts.examples_needed import at, measures  # noqa: E402

GRID = [16, 32, 64, 128, 256, 512, 1024]      # pool sizes every Run 5 setting has
SEEDS = [0, 1, 2, 3, 4, 5]
LRS = {"adamw": [3e-5, 1e-4, 3e-4, 1e-3], "muon": [0.0025, 0.005, 0.01, 0.02]}
RUN2 = {("adamw", 1e-4), ("adamw", 3e-4), ("muon", 0.005), ("muon", 0.01)}   # measured in Run 2
WD = {"adamw": (3e-4, 0.1), "muon": (0.005, 0.006)}                           # matched: lr * wd = 3e-5
TRAINABLE = 1.3
FADE_MS = [64, 128, 256]


def key(opt, lr, wd=0.0):
    return (opt, round(math.log10(lr), 3), round(wd, 6))


def index(runs):
    """{(opt, log10 lr, wd): {(M, seed): run}} for finished runs only."""
    out = {}
    for r in runs:
        if r["done"] is None or r["done"]["status"] != "ok":
            continue
        c = r["cfg"]
        cell = out.setdefault(key(c["opt"], c["lr"], c["wd"]), {})
        assert (c["M"], c["seed"]) not in cell, f"two runs for {c['opt']} {c['lr']} M={c['M']} seed={c['seed']}"
        cell[(c["M"], c["seed"])] = r
    return out


def rel_new(r):
    return tail_mean(r, "mse_ood_last") / r["base"]["ridge_ood_mse_pos"][-1]


def rel_pool(r):
    return tail_mean(r, "mse_id_last") / r["base"]["dmmse_id_mse_pos"][-1]


def trained(r):
    """Post hoc stability check: error on prompts from the run's own pool ends within 1.3x of what
    ridge gets on those same prompts. Any model that trains can reach ridge there (a memoriser
    beats it), so a run above the bound has failed to fit its own training tasks."""
    return tail_mean(r, "mse_id_last") <= TRAINABLE * r["base"]["ridge_id_mse_pos"][-1]


def thresholds(cell):
    """log2 M* per seed on GRID; nan if alpha never crosses 0.5 or a run next to the crossing is missing.

    A seed with a pool size missing still gets a threshold when the two pool sizes that bracket
    the crossing are neighbours in GRID and every smaller pool size is present, because then the
    missing run could not have moved the first downward crossing."""
    out = []
    for s in SEEDS:
        ms = [M for M in GRID if (M, s) in cell]
        al = [tail_mean(cell[(M, s)], "alpha_ood") for M in ms]
        t = math.nan
        for i in range(len(ms) - 1):
            if al[i] >= 0.5 > al[i + 1]:
                if ms[: i + 2] == GRID[: i + 2]:
                    t = threshold_log2M(ms, al)
                break
        out.append(t)
    return np.array(out)


def paired(d):
    """mean, 95% bootstrap CI, sign-flip p, n pairs, n positive."""
    d = np.asarray([v for v in d if np.isfinite(v)])
    if len(d) == 0:
        return dict(mean=math.nan, lo=math.nan, hi=math.nan, p=math.nan, n=0, pos=0)
    p, n = sign_flip_p(d)
    lo, hi = boot_ci(d)
    return dict(mean=float(d.mean()), lo=lo, hi=hi, p=p, n=n, pos=int((d > 0).sum()))


def fmt_lr(lr):
    return f"{lr:g}"


def name(opt, lr, wd=0.0):
    return f"{'AdamW' if opt == 'adamw' else 'Muon'} {fmt_lr(lr)}" + (f" + decay {wd:g}" if wd else "")


def main():
    lr_dir, wd_dir, report, figs = sys.argv[1:5]
    os.makedirs(figs, exist_ok=True)
    raw_lr, raw_wd = load(lr_dir), load(wd_dir)
    grid = index(raw_lr)
    wdx = index(raw_wd)
    L = []
    w = L.append

    settings = [(o, lr) for o in ("adamw", "muon") for lr in LRS[o]]
    cells = {(o, lr): grid.get(key(o, lr), {}) for o, lr in settings}
    wcells = {o: wdx.get(key(o, *WD[o]), {}) for o in WD}

    # ---------- 0. completeness ----------
    w("# Run 5: robustness to learning rate and weight decay\n")
    w("Written by `scripts/run05_robustness.py` from the saved metrics. The rules applied here were")
    w("fixed in `PLAN.md` (\"Run 5 design\") before any Run 5 run had finished. A run's end value is")
    w("the mean of its last 5 evaluations. Six seeds; seed s uses the same task pool, starting")
    w("weights and batches in every setting, so every comparison is paired.\n")
    w("## Runs found\n")
    w("| Setting | From | Finished runs (of 48) |")
    w("|---|---|---|")
    complete, n_found = {}, {}
    for o, lr in settings:
        cell = cells[(o, lr)]
        n_found[(o, lr)] = sum((M, s) in cell for M in GRID + [0] for s in SEEDS)
        # ready for the rules: every infinite-pool run is in, and each seed's threshold is settled
        t = thresholds(cell)
        complete[(o, lr)] = all((0, s) in cell for s in SEEDS) and all(
            all((M, s) in cell for M in GRID) or np.isfinite(t[i]) for i, s in enumerate(SEEDS))
        w(f"| {name(o, lr)} | {'Run 2' if (o, lr) in RUN2 else 'Run 5'} | {n_found[(o, lr)]} |")
    for o in WD:
        n = sum((M, s) in wcells[o] for M in GRID + [0] for s in SEEDS)
        complete[(o, "wd")] = n == 48
        n_found[(o, "wd")] = n
        w(f"| {name(o, *WD[o])} | Run 5b | {n} |")
    w("")
    wanted = {key(o, lr) for o, lr in settings} | {key(o, *WD[o]) for o in WD}
    stopped = [r for r in raw_lr + raw_wd if (r["done"] is None or r["done"]["status"] != "ok")
               and key(r["cfg"]["opt"], r["cfg"]["lr"], r["cfg"]["wd"]) in wanted
               and (r["cfg"]["M"] in GRID or r["cfg"]["M"] == 0)]
    if all(v == 48 for v in n_found.values()):
        w("All settings are complete.")
    else:
        w("Runs that did not finish (they are left out of every table below):\n")
        for r in stopped:
            c = r["cfg"]
            cell = (wcells[c["opt"]] if c["wd"] else cells[(c["opt"], next(lr for lr in LRS[c["opt"]] if math.isclose(lr, c["lr"])))])
            others = [tail_mean(cell[(c["M"], s)], "alpha_ood") for s in SEEDS if (c["M"], s) in cell]
            w(f"- {name(c['opt'], c['lr'], c['wd'])}, pool size {'infinite' if c['M'] == 0 else c['M']}, seed {c['seed']}: "
              f"stopped at step {r['rows'][-1]['step']:,} of {c['steps']:,}. Over its last 5 evaluations alpha was "
              f"{tail_mean(r, 'alpha_ood'):.2f} and new-task error {rel_new(r):.2f}x ridge; the {len(others)} finished seeds at "
              f"that pool size have alpha {min(others):.2f} to {max(others):.2f}.")
        not_ready = [name(*k) for k, v in complete.items() if not v and k[1] != "wd"]
        w("")
        w("Every setting has the runs the decision rules need: all six infinite-pool runs, and the pool")
        w("sizes on both sides of each seed's threshold." if not not_ready else
          "**Not yet usable by the decision rules:** " + ", ".join(not_ready) + ".")

    # ---------- 1. trainability filter ----------
    w("\n## Which settings count (trainability filter)\n")
    w(f"A setting is used only if its infinite-pool runs end within {TRAINABLE}x of ridge. Reading")
    w("used: every one of the six runs has to be within the bound. The seed mean is shown too.\n")
    w("| Setting | New-task error / ridge with an infinite pool, per seed | Mean | Worst | Trainable? |")
    w("|---|---|---|---|---|")
    trainable = {}
    for k_, cell, label in ([((o, lr), cells[(o, lr)], name(o, lr)) for o, lr in settings]
                            + [((o, "wd"), wcells[o], name(o, *WD[o])) for o in WD]):
        v = [rel_new(cell[(0, s)]) if (0, s) in cell else math.nan for s in SEEDS]
        ok = bool(np.all(np.isfinite(v)) and max(v) <= TRAINABLE)
        trainable[k_] = ok
        verdict = "yes" if ok else ("incomplete" if not np.all(np.isfinite(v)) else "**no**")
        if np.all(np.isfinite(v)) and (np.mean(v) <= TRAINABLE) != ok:
            verdict += " (the seed mean alone would say the opposite)"
        w(f"| {label} | {' '.join(f'{x:.2f}' for x in v)} | {np.nanmean(v):.2f}x | {np.nanmax(v):.2f}x | {verdict} |")

    # ---------- 2. thresholds ----------
    thr = {(o, lr): thresholds(cells[(o, lr)]) for o, lr in settings}
    wthr = {o: thresholds(wcells[o]) for o in WD}
    w("\n## Task-diversity threshold at 30,000 steps\n")
    w("M* is the pool size where a model sits halfway between memorising and regressing")
    w("(alpha = 0.5), from pool sizes 16 to 1,024.\n")
    w("| Setting | M* (tasks) | log2 M*, mean | log2 M* per seed | Used? |")
    w("|---|---|---|---|---|")
    for o, lr in settings:
        t = thr[(o, lr)]
        used = "yes" if trainable[(o, lr)] and complete[(o, lr)] else ("no: untrainable" if complete[(o, lr)] else "no: incomplete")
        w(f"| {name(o, lr)} | {2 ** np.nanmean(t):.0f} | {np.nanmean(t):.2f} | {' '.join(f'{v:.2f}' for v in t)} | {used} |")
    for o in WD:
        t = wthr[o]
        w(f"| {name(o, *WD[o])} | {2 ** np.nanmean(t):.0f} | {np.nanmean(t):.2f} | {' '.join(f'{v:.2f}' for v in t)} | {'yes' if trainable[(o, 'wd')] else 'no'} |")

    # ---------- 3. decision rule 1 ----------
    def rule1(usable, label):
        if not (usable["adamw"] and usable["muon"]):
            w("Not enough settings to apply the rule.")
            return
        w("Settings in the comparison: " + ", ".join(name(o, lr) for o in LRS for lr in usable[o]) + ".\n")
        a_hi = max(usable["adamw"], key=lambda lr: np.nanmean(thr[("adamw", lr)]))
        m_lo = min(usable["muon"], key=lambda lr: np.nanmean(thr[("muon", lr)]))
        d = thr[("muon", m_lo)] - thr[("adamw", a_hi)]
        st = paired(d)
        survives = st["n"] == 6 and st["pos"] == 6
        w(f"- Muon's lowest threshold: **{name('muon', m_lo)}**, {2 ** np.nanmean(thr[('muon', m_lo)]):.0f} tasks.")
        w(f"- AdamW's highest threshold: **{name('adamw', a_hi)}**, {2 ** np.nanmean(thr[('adamw', a_hi)]):.0f} tasks.")
        w(f"- Muon minus AdamW, per seed (doublings): {' '.join(f'{v:+.2f}' for v in d)}.")
        w(f"- Mean {st['mean']:+.2f} doublings ({2 ** st['mean']:.2f}x), 95% CI [{st['lo']:+.2f}, {st['hi']:+.2f}], "
          f"Muon above AdamW in {st['pos']} of {st['n']} seeds, p = {st['p']:.3f}.")
        w(f"- **Verdict: the gap {'survives' if survives else 'does NOT survive'} tuning {label}.**")

    usable = {o: [lr for lr in LRS[o] if trainable[(o, lr)] and complete[(o, lr)]] for o in LRS}
    w("\n## Decision rule 1: does the threshold gap survive learning-rate tuning?\n")
    w("Rule: among trainable settings, take Muon's lowest-threshold learning rate and AdamW's")
    w("highest-threshold learning rate. The gap survives if Muon is still above AdamW in 6 of 6")
    w("seeds (paired sign-flip p = 0.031).\n")
    w("### As written\n")
    pending = [name(o, lr) for o, lr in settings if not complete[(o, lr)]]
    if pending:
        w("**The rule cannot be applied as written yet**: " + ", ".join(pending) + " has runs still to finish, and")
        w("the rule ranges over every trainable setting. The figures below use the complete settings")
        w("only and are provisional.\n")
        rule1(usable, "over the complete settings (provisional)")
    else:
        rule1(usable, "by the rule as written")

    # ---- post hoc: stability at every pool size ----
    w("\n### Added after looking: does each setting train at every pool size?\n")
    w("The filter above only looks at the infinite pool. This table counts, per pool size, the runs")
    w(f"whose error on prompts from their own pool ends above {TRAINABLE}x what ridge gets on those same")
    w("prompts. Any model that trains can reach ridge there, and a memoriser beats it, so such a")
    w("run has failed to fit its own training tasks. Its alpha then says nothing about memorising")
    w("against regressing.\n")
    w("| Setting | " + " | ".join(str(M) for M in GRID) + " | infinite | Trains everywhere? |")
    w("|---|" + "---|" * (len(GRID) + 2))
    stable = {}
    for k_, cell, label in ([((o, lr), cells[(o, lr)], name(o, lr)) for o, lr in settings]
                            + [((o, "wd"), wcells[o], name(o, *WD[o])) for o in WD]):
        row, bad = [], 0
        for M in GRID + [0]:
            rs = [cell[(M, s)] for s in SEEDS if (M, s) in cell]
            nb = sum(not trained(r) for r in rs)
            bad += nb
            row.append(f"{nb} of {len(rs)}" if nb else f"0 of {len(rs)}")
        stable[k_] = bad == 0
        w(f"| {label} | " + " | ".join(row) + f" | {'yes' if bad == 0 else '**no**'} |")
    for k_, cell, label in [((o, lr), cells[(o, lr)], name(o, lr)) for o, lr in settings if not stable[(o, lr)]]:
        w(f"\nWhat the failed runs of {label} did: error on their own pool, relative to the best possible")
        w("memoriser (dMMSE), at its lowest point after step 300 and at the end (median over the seeds finished).\n")
        w("| Pool size | Runs | Lowest pool-task error / dMMSE | At step | Final pool-task error / dMMSE | Final new-task error / ridge |")
        w("|---|---|---|---|---|---|")
        for M in GRID:
            rs = [cell[(M, s)] for s in SEEDS if (M, s) in cell]
            if not rs or all(trained(r) for r in rs):
                continue
            lows, steps_, fins = [], [], []
            for r in rs:
                dm = r["base"]["dmmse_id_mse_pos"][-1]
                pts = [(row["mse_id_last"] / dm, row["step"]) for row in r["rows"] if row["step"] >= 300]
                lo_, st_ = min(pts)
                lows.append(lo_)
                steps_.append(st_)
                fins.append(rel_pool(r))
            w(f"| {M} | {len(rs)} | {np.median(lows):.2f}x | {np.median(steps_):,.0f} | {np.median(fins):.2f}x | "
              f"{np.median([rel_new(r) for r in rs]):.2f}x |")
        t = thr[k_]
        clean = []
        for i, sd in enumerate(SEEDS):
            if not np.isfinite(t[i]):
                continue
            lo_M = max(M for M in GRID if math.log2(M) <= t[i])
            hi_M = min(M for M in GRID if math.log2(M) > t[i])
            if trained(cell[(lo_M, sd)]) and trained(cell[(hi_M, sd)]):
                clean.append(i)
        w(f"\nIn {len(clean)} of {len(SEEDS)} seeds the threshold of {label} lies between two pool sizes whose runs trained "
          f"(log2 M* {' '.join(f'{t[i]:.2f}' for i in clean)}). In the other {len(SEEDS) - len(clean)} it lies next to a run that failed.")
    usable2 = {o: [lr for lr in usable[o] if stable[(o, lr)]] for o in LRS}
    w("\n### Added after looking: the same rule over settings that train at every pool size\n")
    rule1(usable2, "among settings that train at every pool size")
    w("\nEvery pairing (Muon minus AdamW, in doublings of M*; seeds with Muon above; p):\n")
    w("| | " + " | ".join(name("muon", lr) for lr in LRS["muon"]) + " |")
    w("|---|" + "---|" * len(LRS["muon"]))
    for la in LRS["adamw"]:
        row = []
        for lm in LRS["muon"]:
            st = paired(thr[("muon", lm)] - thr[("adamw", la)])
            flag = "" if (trainable[("adamw", la)] and trainable[("muon", lm)]) else " †"
            flag += "" if (stable[("adamw", la)] and stable[("muon", lm)]) else " ‡"
            row.append(f"{st['mean']:+.2f}, {st['pos']}/{st['n']}, p={st['p']:.3f}{flag}" if st["n"] else "no pairs")
        w(f"| {name('adamw', la)} | " + " | ".join(row) + " |")
    w("\n† one of the two settings failed the trainability filter. ‡ one of the two does not train at every")
    w("pool size (table above), so its threshold is not a clean reading.\n")
    w("Within each optimizer, one learning-rate step (paired, in doublings of M*):\n")
    w("| Step | Change in M* | Seeds that rise | p |")
    w("|---|---|---|---|")
    for o in LRS:
        for lo_, hi_ in zip(LRS[o][:-1], LRS[o][1:]):
            st = paired(thr[(o, hi_)] - thr[(o, lo_)])
            mark = "" if (stable[(o, lo_)] and stable[(o, hi_)]) else " ‡"
            w(f"| {name(o, lo_)} -> {fmt_lr(hi_)} | {st['mean']:+.2f}{mark} | {st['pos']}/{st['n']} | {st['p']:.3f} |")

    # ---- post hoc: unstable Muon settings against AdamW's top learning rate, where both train ----
    unstable = [lr for lr in LRS["muon"] if not stable[("muon", lr)]]
    if unstable and usable["adamw"]:
        a_hi = max(usable["adamw"], key=lambda lr: np.nanmean(thr[("adamw", lr)]))
        for lm in unstable:
            cm, ca = cells[("muon", lm)], cells[("adamw", a_hi)]
            w(f"\n### Added after looking: {name('muon', lm)} against {name('adamw', a_hi)}, pool size by pool size\n")
            w("alpha on new tasks (1 = memorising, 0 = regressing), Muon minus AdamW, paired by seed.")
            w("\"Runs that failed\" counts Muon runs that did not fit their own pool (table above).\n")
            w("| Pool size | Muon runs that failed | alpha, AdamW / Muon | Difference [95% CI] | Seeds with Muon more memorising | p |")
            w("|---|---|---|---|---|---|")
            for M in GRID:
                ss = [s for s in SEEDS if (M, s) in cm and (M, s) in ca]
                if not ss:
                    w(f"| {M} | no finished pairs | | | | |")
                    continue
                am = np.array([tail_mean(cm[(M, s)], "alpha_ood") for s in ss])
                aa = np.array([tail_mean(ca[(M, s)], "alpha_ood") for s in ss])
                nb = sum(not trained(cm[(M, s)]) for s in ss)
                st = paired(am - aa)
                w(f"| {M} | {nb} of {len(ss)} | {aa.mean():.2f} / {am.mean():.2f} | {st['mean']:+.2f} [{st['lo']:+.2f}, {st['hi']:+.2f}] "
                  f"| {st['pos']}/{st['n']} | {st['p']:.3f} |")

    # ---- post hoc: same ICL onset, different threshold ----
    w("\n### Added after looking: settings with the same ICL onset still differ in threshold\n")
    w("For each pairing: the Muon / AdamW ratio of ICL onset steps (geometric mean over pool sizes")
    w("32 to infinite, then over seeds; 1.0 = same onset), next to the threshold gap from the table above.\n")
    w("| Pairing | Onset ratio, Muon / AdamW [95% CI] | Seeds with Muon sooner | p | Threshold gap (doublings) |")
    w("|---|---|---|---|---|")
    for la, lm in ((1e-3, 0.0025), (3e-4, 0.0025), (3e-4, 0.005), (1e-4, 0.005)):
        ca, cm = cells[("adamw", la)], cells[("muon", lm)]
        per_seed = []
        for s in SEEDS:
            v = []
            for M in GRID[1:] + [0]:
                if (M, s) in ca and (M, s) in cm:
                    oa, om = onset_step(ca[(M, s)]), onset_step(cm[(M, s)])
                    if oa and om:
                        v.append(math.log2(om / oa))
            per_seed.append(np.mean(v) if v else math.nan)
        st = paired(per_seed)
        g = paired(thr[("muon", lm)] - thr[("adamw", la)])
        neg = int(np.sum(np.asarray(per_seed) < 0))
        w(f"| {name('adamw', la)} / {name('muon', lm)} | {2 ** st['mean']:.2f}x [{2 ** st['lo']:.2f}, {2 ** st['hi']:.2f}] "
          f"| {neg}/{st['n']} | {st['p']:.3f} | {g['mean']:+.2f} ({g['pos']}/{g['n']} seeds) |")

    # ---------- 4. decision rule 2 ----------
    w("\n## Decision rule 2: does the gap survive weight decay?\n")
    w("Both optimizers shrink the block matrices by the same factor per step (lr x decay = 3e-5):")
    w(f"{name('adamw', *WD['adamw'])} against {name('muon', *WD['muon'])}.\n")
    d = wthr["muon"] - wthr["adamw"]
    st = paired(d)
    w(f"- Muon minus AdamW, per seed (doublings): {' '.join(f'{v:+.2f}' for v in d)}.")
    w(f"- Mean {st['mean']:+.2f} doublings ({2 ** st['mean']:.2f}x), 95% CI [{st['lo']:+.2f}, {st['hi']:+.2f}], "
      f"Muon above AdamW in {st['pos']} of {st['n']} seeds, p = {st['p']:.3f}.")
    ok2 = st["n"] == 6 and st["pos"] == 6 and trainable[("adamw", "wd")] and trainable[("muon", "wd")]
    w(f"- **Verdict: the gap {'survives' if ok2 else 'does NOT survive'} matched weight decay.**")
    w("\nWhat decay does to each optimizer's own threshold (with decay minus without, same seed):\n")
    w("| Optimizer | M* without / with decay | Change (doublings) | Seeds that rise | p |")
    w("|---|---|---|---|---|")
    for o in WD:
        base = thr[(o, WD[o][0])]
        st = paired(wthr[o] - base)
        w(f"| {name(o, WD[o][0])} | {2 ** np.nanmean(base):.0f} / {2 ** np.nanmean(wthr[o]):.0f} | {st['mean']:+.2f} "
          f"[{st['lo']:+.2f}, {st['hi']:+.2f}] | {st['pos']}/{st['n']} | {st['p']:.3f} |")

    # ---------- 5. rule 3: fading ----------
    w("\n## Rule 3: weight decay and fading (reported as measured, no pass/fail)\n")
    w("Fade is final new-task error divided by the best new-task error the run reached (1.0 = no")
    w("fading). \"Change\" is with decay minus without, paired by seed.\n")
    w("| Pool size | Optimizer | Fade without / with decay | Change [95% CI] | Seeds that fade more | p | Final new-task error vs ridge, without / with |")
    w("|---|---|---|---|---|---|---|")
    for M in FADE_MS:
        for o in WD:
            base, dec = cells[(o, WD[o][0])], wcells[o]
            ss = [s for s in SEEDS if (M, s) in base and (M, s) in dec]
            f0 = np.array([fade(base[(M, s)])[2] for s in ss])
            f1 = np.array([fade(dec[(M, s)])[2] for s in ss])
            st = paired(f1 - f0)
            e0 = np.mean([rel_new(base[(M, s)]) for s in ss])
            e1 = np.mean([rel_new(dec[(M, s)]) for s in ss])
            w(f"| {M} | {name(o, WD[o][0])} | {f0.mean():.2f} / {f1.mean():.2f} | {st['mean']:+.2f} [{st['lo']:+.2f}, {st['hi']:+.2f}] "
              f"| {st['pos']}/{st['n']} | {st['p']:.3f} | {e0:.2f}x / {e1:.2f}x |")

    # ---------- 6. rule 4: onset and examples ----------
    w("\n## Rule 4: ICL onset and examples needed in the new cells (reported, no decision)\n")
    w("ICL onset step, median of 6 seeds:\n")
    allcols = [(name(o, lr), cells[(o, lr)]) for o, lr in settings] + [(name(o, *WD[o]), wcells[o]) for o in WD]
    w("| Pool size | " + " | ".join(n for n, _ in allcols) + " |")
    w("|---|" + "---|" * len(allcols))
    for M in GRID + [0]:
        row = []
        for _, cell in allcols:
            v = [onset_step(cell[(M, s)]) for s in SEEDS if (M, s) in cell]
            v = [x for x in v if x is not None]
            row.append(f"{np.median(v):.0f}" if v else "never")
        w(f"| {'infinite' if M == 0 else M} | " + " | ".join(row) + " |")
    w("\nRidge-equivalent examples (out of 15, mean of 6 seeds): with 15 examples in the prompt, the")
    w("model is as accurate as optimal regression with this many.\n")
    w("| Pool size | " + " | ".join(n for n, _ in allcols) + " |")
    w("|---|" + "---|" * len(allcols))
    for M in GRID + [0]:
        row = []
        for _, cell in allcols:
            v = [measures(at(cell[(M, s)], slice(-5, None)), cell[(M, s)]["base"]["ridge_ood_mse_pos"])[2]
                 for s in SEEDS if (M, s) in cell]
            row.append(f"{np.mean(v):.1f}" if v else "-")
        w(f"| {'infinite' if M == 0 else M} | " + " | ".join(row) + " |")
    w("\nNew-task error vs ridge / pool-task error vs dMMSE at the end (mean of 6 seeds):\n")
    w("| Pool size | " + " | ".join(n for n, _ in allcols) + " |")
    w("|---|" + "---|" * len(allcols))
    for M in GRID + [0]:
        row = []
        for _, cell in allcols:
            rs = [cell[(M, s)] for s in SEEDS if (M, s) in cell]
            row.append(f"{np.mean([rel_new(r) for r in rs]):.2f}x / {np.mean([rel_pool(r) for r in rs]):.2f}x" if rs else "-")
        w(f"| {'infinite' if M == 0 else M} | " + " | ".join(row) + " |")

    open(report, "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    # ---------- figure 1: threshold against learning rate ----------
    ink, muted = "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "text.color": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, o in zip(axes, ("adamw", "muon")):
        xs = LRS[o]
        good = [lr for lr in xs if stable[(o, lr)] and trainable[(o, lr)]]
        ax.plot(good, [np.nanmean(thr[(o, lr)]) for lr in good], color=COL[o], lw=2, zorder=2)
        for lr in xs:
            ok = lr in good
            t = thr[(o, lr)]
            ax.scatter([lr] * len(t), t, s=14, color=COL[o], alpha=0.4, facecolors=COL[o] if ok else "none", zorder=3)
            ax.scatter([lr], [np.nanmean(t)], s=80, color=COL[o], facecolors=COL[o] if ok else "white",
                       edgecolor="white" if ok else COL[o], linewidth=2, zorder=4,
                       label=("no weight decay: mean (large) and seeds (small)" if lr == xs[0] else None))
            if not ok:
                ax.annotate("does not train stably\non pools of 16 and 32 tasks", (lr, np.nanmean(t)), xytext=(-8, 14),
                            textcoords="offset points", ha="right", fontsize=7.5, color=ink)
        lr_wd, wd_ = WD[o]
        ax.scatter([lr_wd * 1.13], [np.nanmean(wthr[o])], s=60, marker="s", facecolors="white", edgecolor=ink,
                   linewidth=1.5, zorder=4, label=f"with weight decay {wd_:g} (mean)")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([fmt_lr(x) for x in xs])
        ax.minorticks_off()
        ax.set_xlabel(f"{'AdamW' if o == 'adamw' else 'Muon'} learning rate (attention and MLP matrices)")
        ax.set_title("AdamW" if o == "adamw" else "Muon", fontsize=10, loc="left")
        ax.legend(fontsize=7.5, loc="lower right" if o == "adamw" else "lower left")
        ax.grid(alpha=0.18, lw=0.6)
    ticks = [16, 24, 32, 48, 64, 96]
    axes[0].set_yticks([math.log2(t) for t in ticks])
    axes[0].set_yticklabels([str(t) for t in ticks])
    axes[0].set_ylim(math.log2(14), math.log2(110))
    axes[0].set_ylabel("threshold M* after 30,000 steps (tasks)")
    fig.suptitle("AdamW's threshold climbs with its learning rate; Muon's stays flat until it stops training stably", fontsize=9,
                 color=muted, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "threshold_vs_lr.png"), dpi=150)
    plt.close(fig)

    # ---------- figure 2: alpha against pool size, every setting ----------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    shades = [0.3, 0.55, 0.8, 1.0]
    for o in ("adamw", "muon"):
        for lr, a in zip(LRS[o], shades):
            cell = cells[(o, lr)]
            mu = [np.mean([tail_mean(cell[(M, s)], "alpha_ood") for s in SEEDS if (M, s) in cell] or [np.nan]) for M in GRID]
            axes[0].plot(GRID, mu, color=COL[o], alpha=a, marker="o", ms=3, lw=1.3,
                         ls="-" if stable[(o, lr)] else ":", label=name(o, lr))
    for o in WD:
        for wd_on, ls in ((False, "-"), (True, "--")):
            cell = wcells[o] if wd_on else cells[(o, WD[o][0])]
            mu = [np.mean([tail_mean(cell[(M, s)], "alpha_ood") for s in SEEDS if (M, s) in cell] or [np.nan]) for M in GRID]
            axes[1].plot(GRID, mu, color=COL[o], marker="o", ms=3, lw=1.3, ls=ls,
                         label=name(o, *WD[o]) if wd_on else name(o, WD[o][0]))
    for ax, t in zip(axes, ("every learning rate, no weight decay (dotted: unstable on small pools)", "with and without matched weight decay")):
        ax.axhline(0.5, color="k", ls=":", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(GRID)
        ax.set_xticklabels([str(m) for m in GRID], fontsize=7)
        ax.set_xlabel("task pool size M")
        ax.set_title(t, fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("alpha on new tasks (1 = memorising, 0 = regressing)")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "alpha_vs_pool_size.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
