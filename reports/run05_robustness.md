# Run 5: robustness to learning rate and weight decay

Written by `scripts/run05_robustness.py` from the saved metrics. The rules applied here were
fixed in `PLAN.md` ("Run 5 design") before any Run 5 run had finished. A run's end value is
the mean of its last 5 evaluations. Six seeds; seed s uses the same task pool, starting
weights and batches in every setting, so every comparison is paired.

## Runs found

| Setting | From | Finished runs (of 48) |
|---|---|---|
| AdamW 3e-05 | Run 5 | 48 |
| AdamW 0.0001 | Run 2 | 48 |
| AdamW 0.0003 | Run 2 | 48 |
| AdamW 0.001 | Run 5 | 48 |
| Muon 0.0025 | Run 5 | 48 |
| Muon 0.005 | Run 2 | 48 |
| Muon 0.01 | Run 2 | 48 |
| Muon 0.02 | Run 5 | 48 |
| AdamW 0.0003 + decay 0.1 | Run 5b | 48 |
| Muon 0.005 + decay 0.006 | Run 5b | 48 |

All settings are complete.

## Which settings count (trainability filter)

A setting is used only if its infinite-pool runs end within 1.3x of ridge. Reading
used: every one of the six runs has to be within the bound. The seed mean is shown too.

| Setting | New-task error / ridge with an infinite pool, per seed | Mean | Worst | Trainable? |
|---|---|---|---|---|
| AdamW 3e-05 | 1.09 1.09 1.11 1.09 1.10 1.13 | 1.10x | 1.13x | yes |
| AdamW 0.0001 | 1.08 1.08 1.09 1.09 1.07 1.09 | 1.08x | 1.09x | yes |
| AdamW 0.0003 | 1.10 1.07 1.13 1.09 1.10 1.10 | 1.10x | 1.13x | yes |
| AdamW 0.001 | 1.09 1.09 1.11 1.09 1.08 1.08 | 1.09x | 1.11x | yes |
| Muon 0.0025 | 1.06 1.05 1.08 1.06 1.06 1.06 | 1.06x | 1.08x | yes |
| Muon 0.005 | 1.08 1.08 1.08 1.06 1.07 1.07 | 1.07x | 1.08x | yes |
| Muon 0.01 | 1.09 1.09 1.09 1.10 1.09 1.08 | 1.09x | 1.10x | yes |
| Muon 0.02 | 1.16 1.23 1.16 1.14 1.16 1.21 | 1.18x | 1.23x | yes |
| AdamW 0.0003 + decay 0.1 | 1.12 1.08 1.16 1.10 1.11 1.11 | 1.11x | 1.16x | yes |
| Muon 0.005 + decay 0.006 | 1.11 1.09 1.10 1.08 1.08 1.09 | 1.09x | 1.11x | yes |

## Task-diversity threshold at 30,000 steps

M* is the pool size where a model sits halfway between memorising and regressing
(alpha = 0.5), from pool sizes 16 to 1,024.

| Setting | M* (tasks) | log2 M*, mean | log2 M* per seed | Used? |
|---|---|---|---|---|
| AdamW 3e-05 | 23 | 4.55 | 4.58 4.38 4.66 4.85 4.39 4.42 | yes |
| AdamW 0.0001 | 28 | 4.79 | 5.04 4.59 4.85 5.05 4.53 4.69 | yes |
| AdamW 0.0003 | 41 | 5.35 | 5.43 5.25 5.35 5.56 5.25 5.25 | yes |
| AdamW 0.001 | 49 | 5.62 | 5.58 5.61 5.66 5.85 5.46 5.56 | yes |
| Muon 0.0025 | 77 | 6.27 | 6.34 6.30 6.26 6.37 6.10 6.26 | yes |
| Muon 0.005 | 80 | 6.33 | 6.30 6.26 6.33 6.39 6.39 6.30 | yes |
| Muon 0.01 | 77 | 6.27 | 6.26 6.21 6.27 6.44 6.06 6.39 | yes |
| Muon 0.02 | 40 | 5.34 | 6.03 4.35 4.72 6.37 4.50 6.07 | yes |
| AdamW 0.0003 + decay 0.1 | 39 | 5.29 | 5.41 5.22 5.26 5.47 5.21 5.19 | yes |
| Muon 0.005 + decay 0.006 | 80 | 6.32 | 6.34 6.28 6.29 6.37 6.39 6.28 | yes |

## Decision rule 1: does the threshold gap survive learning-rate tuning?

Rule: among trainable settings, take Muon's lowest-threshold learning rate and AdamW's
highest-threshold learning rate. The gap survives if Muon is still above AdamW in 6 of 6
seeds (paired sign-flip p = 0.031).

### As written

Settings in the comparison: AdamW 3e-05, AdamW 0.0001, AdamW 0.0003, AdamW 0.001, Muon 0.0025, Muon 0.005, Muon 0.01, Muon 0.02.

- Muon's lowest threshold: **Muon 0.02**, 40 tasks.
- AdamW's highest threshold: **AdamW 0.001**, 49 tasks.
- Muon minus AdamW, per seed (doublings): +0.45 -1.27 -0.94 +0.52 -0.96 +0.51.
- Mean -0.28 doublings (0.82x), 95% CI [-0.87, +0.26], Muon above AdamW in 3 of 6 seeds, p = 0.500.
- **Verdict: the gap does NOT survive tuning by the rule as written.**

### Added after looking: does each setting train at every pool size?

The filter above only looks at the infinite pool. This table counts, per pool size, the runs
whose error on prompts from their own pool ends above 1.3x what ridge gets on those same
prompts. Any model that trains can reach ridge there, and a memoriser beats it, so such a
run has failed to fit its own training tasks. Its alpha then says nothing about memorising
against regressing.

| Setting | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | infinite | Trains everywhere? |
|---|---|---|---|---|---|---|---|---|---|
| AdamW 3e-05 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | yes |
| AdamW 0.0001 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | yes |
| AdamW 0.0003 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | yes |
| AdamW 0.001 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | yes |
| Muon 0.0025 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | yes |
| Muon 0.005 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | yes |
| Muon 0.01 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | yes |
| Muon 0.02 | 4 of 6 | 6 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | **no** |
| AdamW 0.0003 + decay 0.1 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | yes |
| Muon 0.005 + decay 0.006 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | 0 of 6 | yes |

What the failed runs of Muon 0.02 did: error on their own pool, relative to the best possible
memoriser (dMMSE), at its lowest point after step 300 and at the end (median over the seeds finished).

| Pool size | Runs | Lowest pool-task error / dMMSE | At step | Final pool-task error / dMMSE | Final new-task error / ridge |
|---|---|---|---|---|---|
| 16 | 6 | 1.26x | 2,442 | 5.11x | 15.00x |
| 32 | 6 | 1.18x | 9,415 | 7.78x | 12.41x |

In 3 of 6 seeds the threshold of Muon 0.02 lies between two pool sizes whose runs trained (log2 M* 6.03 6.37 6.07). In the other 3 it lies next to a run that failed.

### Added after looking: the same rule over settings that train at every pool size

Settings in the comparison: AdamW 3e-05, AdamW 0.0001, AdamW 0.0003, AdamW 0.001, Muon 0.0025, Muon 0.005, Muon 0.01.

- Muon's lowest threshold: **Muon 0.0025**, 77 tasks.
- AdamW's highest threshold: **AdamW 0.001**, 49 tasks.
- Muon minus AdamW, per seed (doublings): +0.76 +0.69 +0.60 +0.52 +0.64 +0.70.
- Mean +0.65 doublings (1.57x), 95% CI [+0.59, +0.71], Muon above AdamW in 6 of 6 seeds, p = 0.031.
- **Verdict: the gap survives tuning among settings that train at every pool size.**

Every pairing (Muon minus AdamW, in doublings of M*; seeds with Muon above; p):

| | Muon 0.0025 | Muon 0.005 | Muon 0.01 | Muon 0.02 |
|---|---|---|---|---|
| AdamW 3e-05 | +1.73, 6/6, p=0.031 | +1.78, 6/6, p=0.031 | +1.73, 6/6, p=0.031 | +0.79, 5/6, p=0.062 ‡ |
| AdamW 0.0001 | +1.48, 6/6, p=0.031 | +1.54, 6/6, p=0.031 | +1.48, 6/6, p=0.031 | +0.55, 3/6, p=0.250 ‡ |
| AdamW 0.0003 | +0.92, 6/6, p=0.031 | +0.98, 6/6, p=0.031 | +0.92, 6/6, p=0.031 | -0.01, 3/6, p=0.969 ‡ |
| AdamW 0.001 | +0.65, 6/6, p=0.031 | +0.71, 6/6, p=0.031 | +0.65, 6/6, p=0.031 | -0.28, 3/6, p=0.500 ‡ |

† one of the two settings failed the trainability filter. ‡ one of the two does not train at every
pool size (table above), so its threshold is not a clean reading.

Within each optimizer, one learning-rate step (paired, in doublings of M*):

| Step | Change in M* | Seeds that rise | p |
|---|---|---|---|
| AdamW 3e-05 -> 0.0001 | +0.24 | 6/6 | 0.031 |
| AdamW 0.0001 -> 0.0003 | +0.56 | 6/6 | 0.031 |
| AdamW 0.0003 -> 0.001 | +0.27 | 6/6 | 0.031 |
| Muon 0.0025 -> 0.005 | +0.06 | 4/6 | 0.344 |
| Muon 0.005 -> 0.01 | -0.06 | 2/6 | 0.469 |
| Muon 0.01 -> 0.02 | -0.93 ‡ | 0/6 | 0.031 |

### Added after looking: Muon 0.02 against AdamW 0.001, pool size by pool size

alpha on new tasks (1 = memorising, 0 = regressing), Muon minus AdamW, paired by seed.
"Runs that failed" counts Muon runs that did not fit their own pool (table above).

| Pool size | Muon runs that failed | alpha, AdamW / Muon | Difference [95% CI] | Seeds with Muon more memorising | p |
|---|---|---|---|---|---|
| 16 | 4 of 6 | 0.65 / 0.59 | -0.06 [-0.10, -0.04] | 0/6 | 0.031 |
| 32 | 6 of 6 | 0.62 / 0.49 | -0.12 [-0.16, -0.09] | 0/6 | 0.031 |
| 64 | 0 of 6 | 0.43 / 0.52 | +0.09 [+0.07, +0.12] | 6/6 | 0.031 |
| 128 | 0 of 6 | 0.26 / 0.29 | +0.03 [+0.01, +0.04] | 6/6 | 0.031 |
| 256 | 0 of 6 | 0.16 / 0.20 | +0.04 [+0.02, +0.06] | 6/6 | 0.031 |
| 512 | 0 of 6 | 0.07 / 0.07 | +0.00 [-0.02, +0.03] | 3/6 | 0.781 |
| 1024 | 0 of 6 | 0.04 / 0.03 | -0.01 [-0.01, -0.00] | 1/6 | 0.125 |

### Added after looking: settings with the same ICL onset still differ in threshold

For each pairing: the Muon / AdamW ratio of ICL onset steps (geometric mean over pool sizes
32 to infinite, then over seeds; 1.0 = same onset), next to the threshold gap from the table above.

| Pairing | Onset ratio, Muon / AdamW [95% CI] | Seeds with Muon sooner | p | Threshold gap (doublings) |
|---|---|---|---|---|
| AdamW 0.001 / Muon 0.0025 | 0.99x [0.92, 1.06] | 3/6 | 0.750 | +0.65 (6/6 seeds) |
| AdamW 0.0003 / Muon 0.0025 | 0.91x [0.86, 0.96] | 5/6 | 0.062 | +0.92 (6/6 seeds) |
| AdamW 0.0003 / Muon 0.005 | 0.70x [0.66, 0.74] | 6/6 | 0.031 | +0.98 (6/6 seeds) |
| AdamW 0.0001 / Muon 0.005 | 0.52x [0.51, 0.54] | 6/6 | 0.031 | +1.54 (6/6 seeds) |

## Decision rule 2: does the gap survive weight decay?

Both optimizers shrink the block matrices by the same factor per step (lr x decay = 3e-5):
AdamW 0.0003 + decay 0.1 against Muon 0.005 + decay 0.006.

- Muon minus AdamW, per seed (doublings): +0.93 +1.06 +1.03 +0.89 +1.18 +1.09.
- Mean +1.03 doublings (2.04x), 95% CI [+0.95, +1.11], Muon above AdamW in 6 of 6 seeds, p = 0.031.
- **Verdict: the gap survives matched weight decay.**

What decay does to each optimizer's own threshold (with decay minus without, same seed):

| Optimizer | M* without / with decay | Change (doublings) | Seeds that rise | p |
|---|---|---|---|---|
| AdamW 0.0003 | 41 / 39 | -0.05 [-0.08, -0.03] | 0/6 | 0.031 |
| Muon 0.005 | 80 / 80 | -0.01 [-0.03, +0.02] | 3/6 | 0.750 |

## Rule 3: weight decay and fading (reported as measured, no pass/fail)

Fade is final new-task error divided by the best new-task error the run reached (1.0 = no
fading). "Change" is with decay minus without, paired by seed.

| Pool size | Optimizer | Fade without / with decay | Change [95% CI] | Seeds that fade more | p | Final new-task error vs ridge, without / with |
|---|---|---|---|---|---|---|
| 64 | AdamW 0.0003 | 2.43 / 2.34 | -0.10 [-0.17, +0.00] | 1/6 | 0.094 | 5.61x / 5.36x |
| 64 | Muon 0.005 | 2.90 / 2.98 | +0.08 [-0.04, +0.21] | 3/6 | 0.375 | 7.54x / 7.79x |
| 128 | AdamW 0.0003 | 2.11 / 1.94 | -0.17 [-0.22, -0.11] | 0/6 | 0.031 | 3.19x / 2.95x |
| 128 | Muon 0.005 | 2.86 / 2.75 | -0.10 [-0.17, -0.05] | 0/6 | 0.031 | 4.94x / 4.77x |
| 256 | AdamW 0.0003 | 1.34 / 1.26 | -0.08 [-0.12, -0.03] | 1/6 | 0.062 | 1.66x / 1.58x |
| 256 | Muon 0.005 | 2.33 / 2.30 | -0.04 [-0.11, +0.03] | 2/6 | 0.344 | 3.10x / 3.08x |

## Rule 4: ICL onset and examples needed in the new cells (reported, no decision)

ICL onset step, median of 6 seeds:

| Pool size | AdamW 3e-05 | AdamW 0.0001 | AdamW 0.0003 | AdamW 0.001 | Muon 0.0025 | Muon 0.005 | Muon 0.01 | Muon 0.02 | AdamW 0.0003 + decay 0.1 | Muon 0.005 + decay 0.006 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 1744 | 1712 | 1008 | never | 586 | 511 | 390 | 318 | 1008 | 478 |
| 32 | 1500 | 1008 | 824 | 593 | 586 | 446 | 365 | 297 | 768 | 446 |
| 64 | 1262 | 880 | 636 | 586 | 586 | 446 | 340 | 297 | 636 | 446 |
| 128 | 1200 | 824 | 586 | 511 | 586 | 446 | 340 | 297 | 586 | 446 |
| 256 | 1323 | 824 | 586 | 548 | 586 | 446 | 340 | 297 | 586 | 446 |
| 512 | 1412 | 768 | 600 | 593 | 586 | 446 | 390 | 297 | 593 | 446 |
| 1024 | 1412 | 880 | 600 | 628 | 586 | 446 | 390 | 297 | 600 | 446 |
| infinite | 1500 | 768 | 586 | 628 | 586 | 446 | 365 | 297 | 586 | 446 |

Ridge-equivalent examples (out of 15, mean of 6 seeds): with 15 examples in the prompt, the
model is as accurate as optimal regression with this many.

| Pool size | AdamW 3e-05 | AdamW 0.0001 | AdamW 0.0003 | AdamW 0.001 | Muon 0.0025 | Muon 0.005 | Muon 0.01 | Muon 0.02 | AdamW 0.0003 + decay 0.1 | Muon 0.005 + decay 0.006 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 3.0 | 2.4 | 1.8 | 1.2 | 0.8 | 0.9 | 0.2 | 0.4 | 1.8 | 0.9 |
| 32 | 5.7 | 5.0 | 3.3 | 2.6 | 2.6 | 2.1 | 2.0 | 1.9 | 3.4 | 2.0 |
| 64 | 8.1 | 7.4 | 5.7 | 5.0 | 4.8 | 4.4 | 3.8 | 4.5 | 5.9 | 4.2 |
| 128 | 11.5 | 10.1 | 7.9 | 7.9 | 6.6 | 6.2 | 6.6 | 7.0 | 8.2 | 6.3 |
| 256 | 12.9 | 12.5 | 10.9 | 10.0 | 8.0 | 8.0 | 8.3 | 8.6 | 11.1 | 8.0 |
| 512 | 13.3 | 13.4 | 13.2 | 12.6 | 9.4 | 9.6 | 10.3 | 12.2 | 13.1 | 9.8 |
| 1024 | 13.5 | 13.7 | 13.7 | 13.4 | 12.0 | 12.2 | 13.3 | 13.2 | 13.6 | 12.3 |
| infinite | 13.7 | 14.0 | 13.8 | 13.9 | 14.2 | 14.1 | 13.9 | 13.1 | 13.7 | 13.9 |

New-task error vs ridge / pool-task error vs dMMSE at the end (mean of 6 seeds):

| Pool size | AdamW 3e-05 | AdamW 0.0001 | AdamW 0.0003 | AdamW 0.001 | Muon 0.0025 | Muon 0.005 | Muon 0.01 | Muon 0.02 | AdamW 0.0003 + decay 0.1 | Muon 0.005 + decay 0.006 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 9.96x / 1.23x | 10.96x / 1.16x | 11.88x / 1.08x | 12.72x / 1.07x | 13.47x / 1.04x | 13.28x / 1.04x | 14.79x / 1.08x | 14.42x / 7.79x | 11.87x / 1.11x | 13.47x / 1.05x |
| 32 | 5.52x / 1.55x | 6.48x / 1.37x | 9.20x / 1.19x | 10.39x / 1.11x | 10.46x / 1.05x | 11.48x / 1.05x | 11.70x / 1.07x | 11.86x / 10.55x | 9.08x / 1.23x | 11.62x / 1.07x |
| 64 | 2.94x / 2.04x | 3.57x / 1.78x | 5.61x / 1.39x | 6.42x / 1.25x | 6.74x / 1.11x | 7.54x / 1.11x | 8.34x / 1.12x | 7.28x / 1.24x | 5.36x / 1.47x | 7.79x / 1.13x |
| 128 | 1.50x / 2.45x | 1.93x / 2.19x | 3.19x / 1.71x | 3.29x / 1.53x | 4.48x / 1.32x | 4.94x / 1.24x | 4.41x / 1.24x | 3.88x / 1.40x | 2.95x / 1.83x | 4.77x / 1.29x |
| 256 | 1.22x / 2.53x | 1.28x / 2.41x | 1.66x / 2.23x | 2.01x / 1.93x | 3.09x / 1.43x | 3.10x / 1.39x | 2.88x / 1.47x | 2.67x / 1.72x | 1.58x / 2.34x | 3.08x / 1.45x |
| 512 | 1.15x / 2.51x | 1.14x / 2.43x | 1.17x / 2.42x | 1.27x / 2.38x | 2.28x / 1.64x | 2.14x / 1.78x | 1.88x / 1.97x | 1.34x / 2.47x | 1.19x / 2.48x | 2.06x / 1.84x |
| 1024 | 1.13x / 2.49x | 1.10x / 2.41x | 1.10x / 2.42x | 1.14x / 2.47x | 1.37x / 2.15x | 1.35x / 2.20x | 1.16x / 2.38x | 1.17x / 2.55x | 1.12x / 2.46x | 1.33x / 2.24x |
| infinite | 1.10x / 1.12x | 1.08x / 1.10x | 1.10x / 1.10x | 1.09x / 1.10x | 1.06x / 1.07x | 1.07x / 1.08x | 1.09x / 1.09x | 1.18x / 1.18x | 1.11x / 1.12x | 1.09x / 1.10x |
