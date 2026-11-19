Measured on NVIDIA RTX A6000, PyTorch 2.7.1+cu118, batch 256 prompts of 32 tokens, 32-bit floats (Newton-Schulz in bfloat16), median of 200 timed steps.

## Time per training step (milliseconds)

Model = forward + backward. Optimizer = the step for the attention and MLP matrices. The AdamW step for all other parameters is the same in every setting and is listed once per width.

| Optimizer | width 64: model / optimizer / overhead | width 128: model / optimizer / overhead | width 256: model / optimizer / overhead | width 512: model / optimizer / overhead |
|---|---|---|---|---|
| SGD with momentum | 10.4 / 0.43 / 4.2% | 13.0 / 0.12 / 0.9% | 29.3 / 0.40 / 1.4% | 82.8 / 1.58 / 1.9% |
| AdamW | 10.2 / 0.76 / 7.4% | 13.1 / 0.19 / 1.5% | 29.4 / 0.66 / 2.2% | 83.3 / 2.59 / 3.1% |
| Signum | 8.9 / 1.14 / 12.9% | 13.0 / 0.35 / 2.7% | 29.4 / 0.68 / 2.3% | 84.2 / 1.96 / 2.3% |
| Muon, 0 Newton-Schulz steps | 8.9 / 1.86 / 21.0% | 13.0 / 0.60 / 4.6% | 29.3 / 1.08 / 3.7% | 84.6 / 3.10 / 3.7% |
| Muon, 2 Newton-Schulz steps | 8.9 / 2.57 / 28.9% | 13.0 / 0.81 / 6.2% | 29.5 / 1.84 / 6.2% | 84.8 / 6.26 / 7.4% |
| Muon (5 Newton-Schulz steps) | 8.9 / 3.19 / 36.0% | 13.0 / 1.04 / 8.0% | 29.5 / 2.79 / 9.5% | 85.2 / 10.43 / 12.2% |
| Adam direction, Muon step size* | 12.2 / 5.89 / 48.2% | 12.9 / 2.29 / 17.7% | 29.3 / 1.98 / 6.8% | 83.5 / 3.94 / 4.7% |
| AdamW for the other parameters | 0.86 | 0.03 | 0.04 | 0.06 |

## Matrix-product FLOPs per training step (billions)

Counted by PyTorch's FLOP counter, which sees matrix products only; element-wise optimizer work (AdamW, Signum, SGD) reads as zero here and appears in the times above.

| Optimizer | width 64: model / optimizer / overhead | width 128: model / optimizer / overhead | width 256: model / optimizer / overhead | width 512: model / optimizer / overhead |
|---|---|---|---|---|
| SGD with momentum | 21.2 / 0.00 / 0.0% | 81.1 / 0.00 / 0.0% | 316.8 / 0.00 / 0.0% | 1252.1 / 0.00 / 0.0% |
| AdamW | 21.2 / 0.00 / 0.0% | 81.1 / 0.00 / 0.0% | 316.8 / 0.00 / 0.0% | 1252.1 / 0.00 / 0.0% |
| Signum | 21.2 / 0.00 / 0.0% | 81.1 / 0.00 / 0.0% | 316.8 / 0.00 / 0.0% | 1252.1 / 0.00 / 0.0% |
| Muon, 0 Newton-Schulz steps | 21.2 / 0.00 / 0.0% | 81.1 / 0.00 / 0.0% | 316.8 / 0.00 / 0.0% | 1252.1 / 0.00 / 0.0% |
| Muon, 2 Newton-Schulz steps | 21.2 / 0.25 / 1.2% | 81.1 / 2.01 / 2.5% | 316.8 / 16.11 / 5.1% | 1252.1 / 128.85 / 10.3% |
| Muon (5 Newton-Schulz steps) | 21.2 / 0.63 / 3.0% | 81.1 / 5.03 / 6.2% | 316.8 / 40.27 / 12.7% | 1252.1 / 322.12 / 25.7% |
| Adam direction, Muon step size* | 21.2 / 0.00 / 0.0% | 81.1 / 0.00 / 0.0% | 316.8 / 0.00 / 0.0% | 1252.1 / 0.00 / 0.0% |

## Optimizer state memory for the attention and MLP matrices

| Optimizer | width 64 (MB) | width 128 (MB) | width 256 (MB) | width 512 (MB) | bytes per weight |
|---|---|---|---|---|---|
| SGD with momentum | 1.6 | 6.3 | 25.2 | 100.7 | 4 |
| AdamW | 3.1 | 12.6 | 50.3 | 201.3 | 8 |
| Signum | 1.6 | 6.3 | 25.2 | 100.7 | 4 |
| Muon, 0 Newton-Schulz steps | 1.6 | 6.3 | 25.2 | 100.7 | 4 |
| Muon, 2 Newton-Schulz steps | 1.6 | 6.3 | 25.2 | 100.7 | 4 |
| Muon (5 Newton-Schulz steps) | 1.6 | 6.3 | 25.2 | 100.7 | 4 |
| Adam direction, Muon step size* | 3.1 | 12.6 | 50.3 | 201.3 | 8 |

*The Adam-direction control is a per-matrix Python loop written for the ablation, not a fused implementation, so its time reflects that code rather than the method.
