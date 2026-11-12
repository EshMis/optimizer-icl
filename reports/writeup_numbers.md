# Derived figures used in WRITEUP.md

Written by `scripts/writeup_numbers.py`. Do not edit by hand.

## Runs and compute

| Results directory | Steps per run | Finished runs | Wall-clock hours | CPU cores per run | Core-hours |
|---|---|---|---|---|---|
| run01_calibrate | 30,000 | 17 | 25 | 16 | 406 |
| run02_main | 30,000 | 611 | 881 | 8 | 7,050 |
| run03_long | 300,000 | 40 | 537 | 8 | 4,296 |
| run04a_ablation_lr | 30,000 | 14 | 19 | 8 | 152 |
| run05b_wd | 30,000 | 96 | 133 | 8 | 1,062 |

Total: 778 finished training runs, about 13,000 core-hours.

A 30,000-step run on 8 cores took a median of 76 minutes (mean 86, 90th percentile 122; 707 runs).

## Runs as designed (lines in `tasks/`)

- run01_calibrate.txt: 17
- run02_main.txt: 160
- run02_main_s45.txt: 80
- run03_long.txt: 40
- run04_ablation.txt: 120
- run04a_ablation_lr.txt: 14
- run04b_signum_lr.txt: 60
- run05a_lr_threat.txt: 96
- run05b_wd.txt: 96
- run05c_lr_far.txt: 96

Designed in all: 779 runs, of which Run 5 is 288.

## Threshold gap over the learning rates that train at every pool size (30,000 steps)

- 12 pairings (4 AdamW x 3 Muon learning rates). Muon above AdamW in 6 of 6 seeds in every pairing: True.
- Smallest gap 0.65 doublings = 1.57x more tasks; largest 1.78 doublings = 3.44x.
- AdamW thresholds 23 to 49 tasks; Muon thresholds 77 to 80 tasks.
- AdamW 3e-4 against Muon 0.005: 41 against 80 tasks, 0.98 doublings = 1.97x.

## ICL onset in Run 2 (pool sizes 32 to infinite; AdamW 1e-4 or 3e-4 against Muon 0.005 or 0.01)

- Muon's onset is sooner by a factor of 1.3x to 2.8x (geometric mean over 6 seeds, per pairing and pool size; 36 cells). Muon sooner in 6 of 6 seeds in 36 of 36 cells.

## Long runs (Run 3: 300,000 steps, AdamW 3e-4 against Muon 0.005)

- 40 runs: pool sizes 32, 64, 128, 256, 1024; 4 seeds.
- At step 30,000: AdamW's threshold is 43 tasks and Muon's 81 (0.93 doublings, Muon above in 4 of 4 seeds).
- At step 300,000: AdamW's threshold is 127 tasks and Muon's 228 (0.84 doublings, Muon above in 4 of 4 seeds).
- Muon reaches its best new-task error sooner by 2.9x to 4.5x across pool sizes (geometric mean over seeds), and is 10% above its best sooner by 3.8x to 5.2x.
