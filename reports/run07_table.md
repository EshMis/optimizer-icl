## Threshold M* against width (30,000 steps, pools 16 to 256)

| width | params | AdamW M* | per seed (log2) | Muon M* | per seed (log2) | gap (doublings) |
|---|---|---|---|---|---|---|
| 64 | 0.40M | 25 | 4.81 4.50 | 50 | 5.76 5.51 | +0.98 |
| 128 | 1.59M | 41 | 5.43 5.21 5.49 | 80 | 6.35 6.31 6.31 | +0.95 |
| 256 | 6.33M | 57 | 5.88 5.80 | 100 | 6.77 6.52 | +0.81 |
| 512 | 25.24M | 64 | 6.02 5.99 | 124 | 7.21 6.68 | +0.94 |

adamw: M* grows as width^0.45 (slope of log2 M* against log2 width)

muon: M* grows as width^0.43 (slope of log2 M* against log2 width)

## Signum at Muon's Frobenius norm, next to Run 6 (30,000 steps)

| setting | M* | log2 M* per seed |
|---|---|---|
| AdamW 3e-4 | 41 | 5.43 5.21 5.49 |
| Muon, 0 NS steps (normalised momentum) | 41 | 5.39 5.29 5.34 |
| Signum, norm-matched | 51 | 5.67 5.55 5.83 |
| Muon 0.005 | 80 | 6.35 6.31 6.31 |
