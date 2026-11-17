## Threshold M* at 30,000 steps (pools 16 to 256)

| setting | runs | M* (mean) | log2 M* per seed | onset step (median, M=256) | path length, attn / mlp (M=64) | ||W-W0||_F attn / mlp (M=64) |
|---|---|---|---|---|---|---|
| AdamW 3e-4 | 15 | 41 | 5.43 5.21 5.49 | 586 | 7395.2 / 6757.6 | 32.21 / 36.28 |
| Adam direction, Muon step size | 15 | 48 | 5.51 5.54 5.69 | 768 | 53388.0 / 40055.8 | 156.72 / 262.90 |
| Muon, 0 NS steps (no orthogonalisation) | 15 | 41 | 5.39 5.29 5.34 | 1008 | 53393.9 / 40020.0 | 217.50 / 394.24 |
| Muon, 2 NS steps | 15 | 45 | 5.50 5.40 5.58 | 340 | 53393.2 / 40051.3 | 242.20 / 424.27 |
| Muon, 5 NS steps (norm-matched) | 15 | 77 | 6.23 6.15 6.42 | 390 | 53395.0 / 40052.2 | 214.61 / 373.33 |
| Muon 0.005 | 15 | 80 | 6.35 6.31 6.31 | 446 | 37385.4 / 34517.4 | 158.87 / 313.97 |
| Muon on attention only | 15 | 36 | 5.27 4.90 5.29 | 446 | 36131.1 / 6067.2 | 183.29 / 50.64 |
| Muon on MLP only | 15 | 71 | 6.16 5.92 6.39 | 511 | 7080.2 / 34728.3 | 39.35 / 250.97 |

## New-task error / ridge and pool-task error / dMMSE at the end (mean over seeds)

| M | AdamW 3e-4 | Adam direction, Muon step size | Muon, 0 NS steps (no orthogonalisation) | Muon, 2 NS steps | Muon, 5 NS steps (norm-matched) | Muon 0.005 | Muon on attention only | Muon on MLP only |
|---|---|---|---|---|---|---|---|---|
| 16 | 12.06x / 1.10x | 13.14x / 1.06x | 12.99x / 1.09x | 14.53x / 1.15x | 13.05x / 1.05x | 12.82x / 1.04x | 11.03x / 1.11x | 12.76x / 1.05x |
| 32 | 9.31x / 1.16x | 10.00x / 1.10x | 8.67x / 1.15x | 10.39x / 1.14x | 10.67x / 1.07x | 11.08x / 1.05x | 7.52x / 1.21x | 10.72x / 1.06x |
| 64 | 5.45x / 1.39x | 5.89x / 1.16x | 5.12x / 1.38x | 5.65x / 1.20x | 8.29x / 1.11x | 7.56x / 1.11x | 5.18x / 1.38x | 7.97x / 1.10x |
| 128 | 3.11x / 1.68x | 3.63x / 1.50x | 2.41x / 2.22x | 3.50x / 1.53x | 4.54x / 1.24x | 4.84x / 1.22x | 2.98x / 1.68x | 4.68x / 1.33x |
| 256 | 1.62x / 2.15x | 1.69x / 2.09x | 1.40x / 2.75x | 1.72x / 2.28x | 2.97x / 1.42x | 3.05x / 1.35x | 1.69x / 2.15x | 3.19x / 1.46x |

## Paired threshold differences (doublings of M*, per seed)

| setting | minus AdamW 3e-4 | minus Muon 0.005 |
|---|---|---|
| AdamW 3e-4 | +0.00 (+0.00 +0.00 +0.00) | -0.95 (-0.92 -1.10 -0.83) |
| Adam direction, Muon step size | +0.21 (+0.08 +0.33 +0.20) | -0.74 (-0.84 -0.77 -0.62) |
| Muon, 0 NS steps (no orthogonalisation) | -0.03 (-0.04 +0.08 -0.14) | -0.98 (-0.96 -1.02 -0.97) |
| Muon, 2 NS steps | +0.12 (+0.07 +0.19 +0.09) | -0.83 (-0.85 -0.91 -0.73) |
| Muon, 5 NS steps (norm-matched) | +0.89 (+0.80 +0.95 +0.93) | -0.06 (-0.12 -0.16 +0.10) |
| Muon 0.005 | +0.95 (+0.92 +1.10 +0.83) | +0.00 (+0.00 +0.00 +0.00) |
| Muon on attention only | -0.22 (-0.16 -0.31 -0.19) | -1.17 (-1.08 -1.41 -1.02) |
| Muon on MLP only | +0.78 (+0.73 +0.72 +0.90) | -0.17 (-0.19 -0.38 +0.08) |
