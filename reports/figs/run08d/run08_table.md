## Cipher task: threshold M* at 15,000 steps

| setting | runs | M* | log2 M* per seed | onset step (median, unlimited pool) |
|---|---|---|---|---|
| Muon on attention only | 30 | 77 | 6.47 5.93 6.43 | 676 |
| Muon 0.005 | 30 | 217 | 7.82 8.27 7.20 | 1110 |
Full Muon minus Muon-on-attention (effect of Muon on the MLP): +1.48 doublings (+1.35 +2.34 +0.77)

## Per pool size (mean over seeds): alpha on new ciphers | unseen-letter CE on new ciphers minus optimal | unseen-letter accuracy on pool ciphers (memorisation; dMMSE-style optimum in brackets)

| M | Muon on attention only | Muon 0.005 |
|---|---|---|
| 16 | 0.80 | +6.38 | 0.91 (0.91) | 0.84 | +5.89 | 0.91 (0.91) |
| 32 | 0.70 | +4.76 | 0.88 (0.89) | 0.82 | +4.83 | 0.89 (0.89) |
| 64 | 0.55 | +4.36 | 0.86 (0.87) | 0.77 | +3.98 | 0.87 (0.87) |
| 128 | 0.32 | +5.01 | 0.82 (0.85) | 0.62 | +3.63 | 0.85 (0.85) |
| 256 | 0.10 | +4.62 | 0.77 (0.83) | 0.46 | +4.43 | 0.82 (0.83) |
| 512 | 0.04 | +3.13 | 0.69 (0.81) | 0.06 | +2.80 | 0.56 (0.81) |
| 1024 | 0.02 | +1.60 | 0.54 (0.79) | 0.02 | +1.86 | 0.48 (0.79) |
| 2048 | 0.01 | +0.21 | 0.21 (0.78) | 0.01 | +1.46 | 0.51 (0.78) |
| 4096 | 0.00 | +0.01 | 0.13 (0.76) | 0.01 | +0.52 | 0.33 (0.76) |
| unlimited | nan | +0.02 | 0.12 (0.12) | nan | +0.06 | 0.11 (0.12) |
