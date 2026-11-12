"""Per-run fading statistics for the long runs (shared by run03_dynamics.py and writeup_numbers.py)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze import series  # noqa: E402


def fade_stats(r):
    s, y = series(r, "mse_ood_last")
    ridge = r["base"]["ridge_ood_mse_pos"][-1]
    i = int(np.argmin(y))
    final = float(np.mean(y[-5:]))
    after = np.nonzero(y[i:] >= 1.1 * y[i])[0]
    t10 = int(s[i + after[0]]) if len(after) else math.nan
    j150 = int(np.argmin(np.abs(s - 150000)))
    return dict(best=y[i] / ridge, best_step=int(s[i]), fade=final / y[i], t10=t10,
                final=final / ridge, still=final / float(np.mean(y[j150 - 2:j150 + 3])))
