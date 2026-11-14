"""Check every measured number in WRITEUP.md and README.md against the row it came from.

usage: python scripts/check_writeup.py [--file WRITEUP.md=path] [--file README.md=path]

Each claim below ties one sentence (a regular expression over the prose) to the report cells
that must supply its numbers. The reports are the generated files under `reports/`, so this
runs without the raw metrics. A sentence that is reworded, or whose number no longer matches
its source, fails. Any number in the prose that no claim covers also fails, unless it is a
design value listed in DESIGN (pool sizes, learning rates, step counts, model shape).

Exit status: 0 = every claim found and every number accounted for; 1 = a mismatch, a missing
sentence or an uncovered number; 2 = nothing was checked.

`--file NAME=path` checks a different copy of a document (used to test this checker by
altering numbers in a copy and confirming it fails).
"""
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R3 = "reports/run03_dynamics.md"
R4 = "reports/run04_compare.md"
R5 = "reports/run05_robustness.md"
Q3 = "reports/q3_examples_needed_run02.md"
WN = "reports/writeup_numbers.md"

# Design values: chosen, not measured. A number in the prose that is one of these needs no source.
DESIGN = {
    "16", "32", "64", "128", "256", "512", "1024", "2048", "4096",        # pool sizes; 16 pairs; width; batch
    "30000", "300000", "45000",                                            # steps
    "3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "0.0025", "0.005", "0.01", "0.02",  # learning rates
    "8", "2", "1.6", "15", "0.5", "10", "95",                              # model shape, examples, alpha cut, per cent
    "0", "13",                                                             # the range the dropped measure flipped over
    "1", "3", "4", "5", "6",                                               # section numbers, seeds
    "2022", "2023", "2024", "2025", "2026",
}
NUM = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e-?\d+)?(?![\w])")


def read(path):
    return open(os.path.join(ROOT, path)).read()


def prose(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"arXiv:\d{4}\.\d{5}", " ", text)
    return re.sub(r"\s+", " ", text)


def rows(path, after=None):
    """Table rows (lists of cells). With `after`, only the first table below the line containing it;
    without, every table row in the file."""
    out, on = [], after is None
    for line in read(path).splitlines():
        if not on:
            on = after is not None and after in line
            continue
        is_row = line.startswith("|")
        if after is not None and out and not is_row:
            break
        if is_row and not set(line) <= set("|- "):
            out.append([c.strip() for c in line.strip().strip("|").split("|")])
    return out


def row(path, key, after=None):
    for r in rows(path, after):
        if r[0] == key:
            return r
    raise KeyError(f"{path}: no row {key!r} after {after!r}")


def col(path, key, name, after=None):
    """Cell in row `key` under the column whose header is `name`."""
    table = rows(path, after)
    header = next(r for r in table if name in r)
    return row(path, key, after)[header.index(name)]


def grab(path, pattern):
    m = re.search(pattern, re.sub(r"\s+", " ", read(path)))
    if not m:
        raise KeyError(f"{path}: pattern not found: {pattern}")
    return list(m.groups())


def first(cell):                       # "1.93x / 2.19x" -> "1.93"
    return cell.split("/")[0].strip().rstrip("x")


def second(cell):
    return cell.split("/")[1].strip().rstrip("x")


def unsigned(s):                       # "+0.652" -> 0.652
    return abs(float(s))


def f2(x):
    return f"{x:.2f}"


def thr5(setting):
    return row(R5, setting, "## Task-diversity threshold")[1]


def stat(path, label="log2 M*"):
    """(mean diff, p) of a stats.md row."""
    for line in read(path).splitlines():
        if line.startswith("| " + label):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            return c[2], c[4]
    raise KeyError(f"{path}: no row {label}")


STABLE = ["AdamW 3e-05", "AdamW 0.0001", "AdamW 0.0003", "AdamW 0.001", "Muon 0.0025", "Muon 0.005", "Muon 0.01"]


def claims():
    c = []
    add = lambda doc, what, pat, exp: c.append((doc, what, pat, exp))  # noqa: E731

    # ---------- shared look-ups ----------
    means = [float(row(R5, s, "## Which settings count")[2].rstrip("x")) for s in STABLE]
    a_lo, a_hi, m_lo, m_hi = grab(WN, r"AdamW thresholds (\d+) to (\d+) tasks; Muon thresholds (\d+) to (\d+) tasks")
    n_pair, all6 = grab(WN, r"- (\d+) pairings \(4 AdamW x 3 Muon learning rates\)\. Muon above AdamW in 6 of 6 seeds in every pairing: (\w+)")
    assert all6 == "True", "not every pairing has Muon above AdamW in 6 of 6 seeds"
    g_lo, x_lo, g_hi, x_hi = grab(WN, r"Smallest gap ([\d.]+) doublings = ([\d.]+)x more tasks; largest ([\d.]+) doublings = ([\d.]+)x")
    pk_lo, pk_hi, fd_lo, fd_hi = grab(WN, r"sooner by ([\d.]+)x to ([\d.]+)x across pool sizes \(geometric mean over seeds\), and is 10% above its best sooner by ([\d.]+)x to ([\d.]+)x")
    on_lo, on_hi = grab(WN, r"onset is sooner by a factor of ([\d.]+)x to ([\d.]+)x")
    clock = {r[0]: r for r in rows(R3, "## Clock test")[1:]}
    late = [r for k, r in clock.items() if int(k.replace(",", "")) >= 45000]
    per_seed = [float(x) for r in late for x in r[3].split() if x != "nan"]
    near = round(sum(float(r[4]) for r in late) / len(late))
    matrix = rows(R5, "Every pairing (Muon minus AdamW")[1:5]
    assert all("6/6, p=0.031" in cell for r in matrix for cell in r[1:4]), "a stable pairing is not 6/6, p=0.031"
    steps_up = [row(R5, k, "Within each optimizer")[2] for k in ("AdamW 3e-05 -> 0.0001", "AdamW 0.0001 -> 0.0003", "AdamW 0.0003 -> 0.001")]
    assert steps_up == ["6/6"] * 3, "an AdamW learning-rate step does not rise in 6 of 6 seeds"
    paired3 = rows(R3, "## Paired, Muon minus AdamW")[1:]
    assert all(r[2] == "4/4" and r[4] == "4/4" for r in paired3), "Run 3: not 4 of 4 seeds at every pool size"
    rule = "the same rule over settings that train at every pool size"
    m_low = grab(R5, rule + r".*?Muon's lowest threshold: \*\*Muon [\d.]+\*\*, (\d+) tasks")[0]
    a_high = grab(R5, rule + r".*?AdamW's highest threshold: \*\*AdamW [\d.]+\*\*, (\d+) tasks")[0]
    gap, n_pos, n_all, p_gap = grab(R5, rule + r".*?- Mean \+([\d.]+) doublings \([\d.]+x\), 95% CI \[[^\]]*\], Muon above AdamW in (\d) of (\d) seeds, p = ([\d.]+)\.")
    stab = {r[0]: r for r in rows(R5, "does each setting train at every pool size")[1:]}
    others_ok = all(cell.startswith("0 of") for k, r in stab.items() if k != "Muon 0.02" for cell in r[1:9])
    assert others_ok, "a setting other than Muon 0.02 has a run that failed to fit its own pool"
    wd_gap, wd_pos, wd_all = grab(R5, r"## Decision rule 2.*?- Mean \+([\d.]+) doublings \([\d.]+x\), 95% CI \[[^\]]*\], Muon above AdamW in (\d) of (\d) seeds")
    assert (wd_pos, wd_all) == ("6", "6")
    own = [unsigned(row(R5, k, "What decay does to each optimizer")[2].split()[0]) for k in ("AdamW 0.0003", "Muon 0.005")]
    fade_a = next(r for r in rows(R5, "## Rule 3") if r[0] == "128" and r[1] == "AdamW 0.0003")
    fade_m = next(r for r in rows(R5, "## Rule 3") if r[0] == "128" and r[1] == "Muon 0.005")
    assert fade_a[4] == "0/6" and fade_m[4] == "0/6"
    ons = row(R5, "AdamW 0.001 / Muon 0.0025", "settings with the same ICL onset")
    o_ratio, o_lo, o_hi = re.findall(r"([\d.]+)x \[([\d.]+), ([\d.]+)\]", ons[1])[0]
    o_gap = re.findall(r"\+([\d.]+)", ons[4])[0]
    onset_tab = "ICL onset step, median of 6 seeds"
    pools = ["16", "32", "64", "128", "256", "512", "1024", "infinite"]
    on_slow = {col(R5, k, "Muon 0.0025", onset_tab) for k in pools}
    on_fast = [int(col(R5, k, "Muon 0.01", onset_tab)) for k in pools]
    assert len(on_slow) == 1, "Muon 0.0025's median onset is not the same at every pool size"
    k_eq = "**k_equiv**"
    ke = lambda pool, name: [r for r in rows(Q3) if r[0] == pool][2][rows(Q3)[0].index(name)]  # noqa: E731  (third block = k_equiv)
    assert rows(Q3)[[r[0] for r in rows(Q3)].index(k_eq)][0] == k_eq
    big = [r for r in rows(R4, "Error on new tasks vs ridge") if r[0] in ("2048", "4096", "inf")]
    hdr = rows(R4, "Error on new tasks vs ridge")[0]
    worst = max(abs(float(first(r[hdr.index(a)])) / float(first(r[hdr.index(m)])) - 1)
                for r in big for a in ("adamw 0.0001", "adamw 0.0003") for m in ("muon 0.005", "muon 0.01"))
    r128 = next(r for r in rows(R4, "Error on new tasks vs ridge") if r[0] == "128")
    lev = {r[0]: float(r[4]) for r in rows(R3, "## Is either threshold levelling off")[1:]}
    assert all(abs(v - 1.5) <= 0.1 for v in lev.values()), "the threshold's rise per tenfold steps is not about 1.5 doublings"
    f1024 = row(R3, "1024", "## Fade, per pool size")
    sgd_d, sgd_p = stat("reports/figs/run04/adamw_sgdmlr_a1e4/stats.md")
    sig_d, _ = stat("reports/figs/run04/adamw_signumlr_a3e4lr_b3e4/stats.md")
    ms = [unsigned(stat(f"reports/figs/run04/signum_muonlr_{k}/stats.md")[0]) for k in ("b0.005", "b0.01", "a3e4lr_b0.005", "a3e4lr_b0.01")]
    n_done = grab(WN, r"Total: (\d+) finished training runs")[0]
    n5 = grab(WN, r"of which Run 5 is (\d+)")[0]
    minutes = grab(WN, r"A 30,000-step run on 8 cores took a median of (\d+) minutes")[0]
    t300 = grab(WN, r"At step 300,000: AdamW's threshold is (\d+) tasks and Muon's (\d+) \([\d.]+ doublings, Muon above in (\d) of (\d) seeds\)")
    n02 = int(row(R5, "Muon 0.02", "## Runs found")[2])

    # ---------- WRITEUP.md ----------
    W = "WRITEUP.md"
    add(W, "run count", r"Across (\d+) finished training runs", [n_done])
    add(W, "infinite-pool quality", r"between ([\d.]+)x and ([\d.]+)x the best possible error", [f2(min(means)), f2(max(means))])
    add(W, "threshold ranges", r"needs about (\d+) to (\d+) distinct tasks before it does real regression on new tasks\. AdamW needs (\d+) to (\d+), depending", [m_lo, m_hi, a_lo, a_hi])
    add(W, "clock factor (summary)", r"AdamW reaches each threshold about (\d) times later than Muon", [str(near)])
    add(W, "peak sooner (summary)", r"Muon peaks ([\d.]+) to ([\d.]+) times sooner", [pk_lo, pk_hi])
    add(W, "featured thresholds", r"regressing at about (\d+) tasks\. With Muon at 0\.005 the switch comes at about (\d+)\.", [thr5("AdamW 0.0003"), thr5("Muon 0.005")])
    add(W, "errors at 128 tasks", r"AdamW at 1e-4 ends at ([\d.]+)x the optimal error on new tasks and Muon at ([\d.]+)x", [first(r128[hdr.index("adamw 0.0001")]), first(r128[hdr.index("muon 0.005")])])
    for label, key in (("AdamW \\| 3e-5", "AdamW 3e-05"), ("AdamW \\| 1e-4", "AdamW 0.0001"), ("AdamW \\| 3e-4", "AdamW 0.0003"),
                       ("AdamW \\| 1e-3", "AdamW 0.001"), ("Muon \\| 0\\.0025", "Muon 0.0025"), ("Muon \\| 0\\.005", "Muon 0.005"),
                       ("Muon \\| 0\\.01", "Muon 0.01")):
        add(W, f"threshold table: {key}", r"\| " + label + r" \| (\d+) \|", [thr5(key)])
    add(W, "gap over pairings", r"in all (\d+) pairings of these learning rates, by ([\d.]+) to ([\d.]+) doublings \(([\d.]+)x to ([\d.]+)x more tasks\), in 6 of 6 seeds each time \(p = 0\.031\)", [n_pair, g_lo, g_hi, x_lo, x_hi])
    add(W, "large pools", r"the two optimizers end within (\d)% of each other", [str(math.ceil(worst * 100))])
    add(W, "onset", r"sooner: ([\d.]+)x to ([\d.]+)x sooner at every pool size from 32 up", [on_lo, on_hi])
    add(W, "threshold pace", r"about ([\d.]+) doublings for every tenfold increase in steps", ["1.5"])
    for t in ("30,000", "60,000", "150,000", "300,000"):
        r = clock[t]
        add(W, f"clock table: {t}", r"\| " + t + r" \| (\d+) \| ([\d,]+) \| ([\d.]+) \|", [str(round(2 ** float(r[1]))), r[2], r[4]])
    add(W, "clock factor", r"the factor stays near (\d) \(individual seeds range from ([\d.]+) to ([\d.]+)\)", [str(near), f"{min(per_seed):.1f}", f"{max(per_seed):.1f}"])
    add(W, "thresholds at 300,000 steps", r"Muon's threshold is (\d+) tasks against AdamW's (\d+), with Muon higher in (\d) of (\d) seeds", [t300[1], t300[0], t300[2], t300[3]])
    add(W, "fading at 1,024 tasks", r"AdamW gets to ([\d.]+)x the optimal error around step ([\d,]+) and ends at ([\d.]+)x", [first(f1024[1]), first(f1024[2]), first(f1024[4])])
    add(W, "peak and fade factors", r"best error ([\d.]+) to ([\d.]+) times sooner than AdamW and is 10% above its best ([\d.]+) to ([\d.]+) times sooner, in 4 of 4 seeds at every pool size", [pk_lo, pk_hi, fd_lo, fd_hi])
    add(W, "SGD", r"SGD lands on AdamW: (\d+) tasks against (\d+) for AdamW at 1e-4, a difference of ([\d.]+) doublings \(p = ([\d.]+)\)", [row(R4, "sgdm 0.1")[1], row(R4, "adamw 0.0001")[1], f2(unsigned(sgd_d)), sgd_p])
    add(W, "Signum", r"at (\d+) and (\d+) tasks\. At a matched learning rate of 3e-4, Signum is ([\d.]+) doublings above AdamW\. Muon is a further ([\d.]+) to ([\d.]+) doublings above Signum in all four pairings", [row(R4, "signum 0.0001")[1], row(R4, "signum 0.0003")[1], f2(unsigned(sig_d)), f2(min(ms)), f2(max(ms))])
    add(W, "examples, large pools", r"both optimizers score about (\d+) \(([\d.]+) each at 2,048 tasks\)", [str(round(float(ke("2048", "adamw 0.0003")))), ke("2048", "muon 0.005")])
    assert ke("2048", "adamw 0.0003") == ke("2048", "muon 0.005"), "the two optimizers do not score the same at 2,048 tasks"
    add(W, "examples, 256 tasks", r"15 examples are worth ([\d.]+), against ([\d.]+) for AdamW at 3e-4 and ([\d.]+) at 1e-4", [ke("256", "muon 0.005"), ke("256", "adamw 0.0003"), ke("256", "adamw 0.0001")])
    add(W, "Run 5 size", r"A last set of (\d+) runs", [n5])
    add(W, "AdamW learning-rate range", r"from (\d+) tasks at 3e-5 to (\d+) at 1e-3, in 6 of 6 seeds each time", [thr5("AdamW 3e-05"), thr5("AdamW 0.001")])
    add(W, "closest pairing", r"AdamW still switches at (\d+) tasks, against (\d+) for Muon's lowest stable setting: a gap of ([\d.]+) doublings in (\d) of (\d) seeds \(p = ([\d.]+)\)", [a_high, m_low, gap, n_pos, n_all, p_gap])
    add(W, "Muon 0.02 failures", r"(\d of \d) runs at 16 tasks and (\d of \d) at 32 end with an error on their own training tasks more than ([\d.]+)x what ridge regression gets there\. No run at any other setting does that", [stab["Muon 0.02"][1], stab["Muon 0.02"][2], "1.3"])
    lit_m, lit_a, lit_pos, lit_n = (grab(R5, r"### As written.*?Muon's lowest threshold: \*\*Muon 0\.02\*\*, (\d+) tasks")[0],
                                    grab(R5, r"### As written.*?AdamW's highest threshold: \*\*AdamW [\d.]+\*\*, (\d+) tasks")[0],
                                    *grab(R5, r"### As written.*?Muon above AdamW in (\d) of (\d) seeds"))
    assert "does NOT survive tuning by the rule as written" in read(R5), "the report's verdict as written is not 'does NOT survive'"
    add(W, "rule as written", r"Muon's lowest threshold is the (\d+) tasks measured at 0\.02, which is above AdamW's (\d+) in only (\d) of (\d) seeds, and the gap does not survive tuning", [lit_m, lit_a, lit_pos, lit_n])
    n_clean, n_seeds = grab(R5, r"In (\d) of (\d) seeds the threshold of Muon 0\.02 lies between two pool sizes whose runs trained")
    words = {"3": "three", "6": "six"}
    add(W, "seeds with an artefact threshold", r"In (\w+) of the (\w+) seeds these runs pull alpha below 0\.5 at 32 tasks", [words[str(int(n_seeds) - int(n_clean))], words[n_seeds]])
    vs = {r[0]: r for r in rows(R5, "Muon 0.02 against AdamW 0.001, pool size by pool size")[1:]}
    assert all(vs[k][4] == "6/6" for k in ("64", "128", "256")), "Muon 0.02 is not more memorising than AdamW 1e-3 in 6 of 6 seeds at 64, 128 and 256"
    add(W, "Muon 0.02 where it trains", r"still on the memorising side of AdamW at 1e-3, in (\d) of (\d) seeds at 64, 128 and 256 tasks", ["6", "6"])
    assert n02 == 48 and "All settings are complete." in read(R5), "a Run 5 run is unfinished; the write-up says nothing is left out"
    add(W, "weight decay", r"Muon's threshold is (\d+) tasks and AdamW's (\d+), a gap of ([\d.]+) doublings in 6 of 6 seeds", [thr5("Muon 0.005 + decay 0.006"), thr5("AdamW 0.0003 + decay 0.1"), wd_gap])
    add(W, "decay on thresholds", r"Decay moved neither threshold by more than ([\d.]+) doublings", [f2(max(own))])
    add(W, "decay on fading", r"went from ([\d.]+) to ([\d.]+) for AdamW and from ([\d.]+) to ([\d.]+) for Muon", [first(fade_a[2]), second(fade_a[2]), first(fade_m[2]), second(fade_m[2])])
    add(W, "same onset", r"the ratio of their onset steps is ([\d.]+), with a 95% interval of ([\d.]+) to ([\d.]+)\), and Muon's threshold is still ([\d.]+) doublings higher", [o_ratio, o_lo, o_hi, o_gap])
    add(W, "onset against learning rate", r"from about step (\d+) to between (\d+) and (\d+)", [on_slow.pop(), str(min(on_fast)), str(max(on_fast))])
    add(W, "smallest p, 6 seeds", r"6 seeds give a smallest possible p-value of ([\d.]+)", [f"{2 / 2 ** 6:.3f}"])
    add(W, "smallest p, 4 seeds", r"Four seeds make ([\d.]+) the smallest possible p-value", [f"{2 / 2 ** 4:.3f}"])
    add(W, "time per run", r"A 30,000-step run took a median of (\d+) minutes on 8 CPU cores", [minutes])

    # ---------- README.md ----------
    R = "README.md"
    add(R, "threshold ranges", r"it needs (\d+) to (\d+) distinct tasks before in-context learning wins, against (\d+) to (\d+) for AdamW", [m_lo, m_hi, a_lo, a_hi])
    add(R, "clock factor", r"AdamW reaches each threshold about (\d) times later than Muon", [str(near)])
    return c


def main():
    paths = {"WRITEUP.md": "WRITEUP.md", "README.md": "README.md"}
    args = sys.argv[1:]
    while args:
        assert args[0] == "--file", __doc__
        name, path = args[1].split("=", 1)
        paths[name] = path
        args = args[2:]
    texts = {name: prose(read(path)) for name, path in paths.items()}
    try:
        cl = claims()
    except (AssertionError, KeyError, StopIteration, IndexError, AttributeError) as e:
        print(f"SOURCE CHECK FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
    bad, spans = 0, {name: [] for name in texts}
    for doc, what, pat, exp in cl:
        m = re.search(pat, texts[doc])
        if not m:
            print(f"NOT FOUND   {doc}: {what}: sentence missing or reworded")
            bad += 1
            continue
        spans[doc].append(m.span())
        got = list(m.groups())
        if got != exp:
            print(f"MISMATCH    {doc}: {what}: prose says {got}, source says {exp}")
            bad += 1
    loose = 0
    for name, text in texts.items():
        for m in NUM.finditer(text):
            t = m.group(0).replace(",", "")
            if t in DESIGN or any(a <= m.start() and m.end() <= b for a, b in spans[name]):
                continue
            print(f"UNCOVERED   {name}: {m.group(0)!r} in: ...{text[max(0, m.start() - 60):m.end() + 30]}...")
            loose += 1
    if not cl:
        print("NOTHING CHECKED")
        sys.exit(2)
    print(f"{len(cl)} claims checked; {bad} failed; {loose} numbers not covered by any claim")
    sys.exit(1 if bad or loose else 0)


if __name__ == "__main__":
    main()
