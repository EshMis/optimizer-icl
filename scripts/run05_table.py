"""Run 5, one row per run: the four new learning-rate settings (5a, 5c) and the matched-decay runs (5b).

usage: python scripts/run05_table.py results/run02_main results/run05b_wd > reports/run05_table.md

5a and 5c wrote into Run 2's results directory, so their runs are picked out of it by setting.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze import load, table  # noqa: E402

NEW = {"adamw": (3e-5, 1e-3), "muon": (0.0025, 0.02)}   # the settings Run 5 added, no weight decay


def main():
    lr_dir, wd_dir = sys.argv[1:3]
    lr = [r for r in load(lr_dir) if r["cfg"]["wd"] == 0
          and any(math.isclose(r["cfg"]["lr"], x) for x in NEW.get(r["cfg"]["opt"], ()))]
    wd = load(wd_dir)
    unfinished = sum(r["done"] is None for r in lr + wd)
    note = "" if not unfinished else (f" A status of `running` marks a run that did not finish ({unfinished} here);"
                                      " the robustness report leaves those out.")
    print("# Run 5, every run\n")
    print(f"Written by `scripts/run05_table.py` with `analyze.table`. Columns as in `run02_table.md`.{note}\n")
    print(f"## 5a and 5c: learning rates, no weight decay ({len(lr)} runs)\n")
    print(table(lr))
    print(f"\n## 5b: matched weight decay, AdamW 3e-4 with 0.1 and Muon 0.005 with 0.006 ({len(wd)} runs)\n")
    print(table(wd))


if __name__ == "__main__":
    main()
