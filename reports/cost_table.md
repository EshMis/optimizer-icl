Measured on NVIDIA RTX A6000, PyTorch 2.7.1+cu118, batch 256 prompts of 32 tokens, 32-bit floats (Newton-Schulz in bfloat16). Times are the median of 3 independent measurements, each the median of 300 timed steps.

## Everything per training step, in absolute units

One training step = one batch of 256 prompts: the model's forward and backward pass, then the optimizer update. "Extra time" is the optimizer update's time divided by the forward and backward time: 10% means the update adds a tenth on top of the model's own work. Optimizer memory is the state each optimizer keeps for the attention and MLP matrices. Peak GPU memory is the most memory allocated at any point in a step: parameters, gradients, optimizer state, activations and temporaries.

| Width | Parameters | Optimizer | Forward + backward (ms) | Optimizer update (ms) | Update range over repeats (ms) | Extra time | Forward + backward (GFLOPs) | Optimizer update (GFLOPs) | Optimizer memory (MB) | Peak GPU memory (GB) |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 0.4M | SGD with momentum | 9.3 | 0.30 | 0.30 to 0.40 | 3.3% | 21.2 | 0.00 | 1.6 | 0.30 |
| 64 | 0.4M | AdamW | 9.2 | 0.66 | 0.65 to 0.85 | 7.1% | 21.2 | 0.00 | 3.1 | 0.31 |
| 64 | 0.4M | Signum | 9.2 | 1.21 | 1.10 to 1.22 | 13.1% | 21.2 | 0.00 | 1.6 | 0.30 |
| 64 | 0.4M | Muon, 0 Newton-Schulz steps | 9.2 | 1.97 | 1.86 to 1.99 | 21.5% | 21.2 | 0.00 | 1.6 | 0.30 |
| 64 | 0.4M | Muon, 2 Newton-Schulz steps | 9.2 | 2.70 | 2.55 to 2.74 | 29.4% | 21.2 | 0.25 | 1.6 | 0.30 |
| 64 | 0.4M | Muon (5 Newton-Schulz steps) | 9.3 | 3.42 | 3.23 to 3.51 | 36.8% | 21.2 | 0.63 | 1.6 | 0.30 |
| 64 | 0.4M | Adam direction, Muon step size* | 12.5 | 6.07 | 5.63 to 6.17 | 48.6% | 21.2 | 0.00 | 3.1 | 0.31 |
| 128 | 1.6M | SGD with momentum | 13.0 | 0.12 | 0.12 to 0.12 | 0.9% | 81.1 | 0.00 | 6.3 | 0.60 |
| 128 | 1.6M | AdamW | 13.1 | 0.19 | 0.19 to 0.19 | 1.5% | 81.1 | 0.00 | 12.6 | 0.60 |
| 128 | 1.6M | Signum | 13.1 | 0.36 | 0.35 to 0.70 | 2.7% | 81.1 | 0.00 | 6.3 | 0.60 |
| 128 | 1.6M | Muon, 0 Newton-Schulz steps | 13.0 | 0.61 | 0.61 to 1.12 | 4.7% | 81.1 | 0.00 | 6.3 | 0.60 |
| 128 | 1.6M | Muon, 2 Newton-Schulz steps | 13.0 | 2.02 | 1.71 to 2.19 | 15.5% | 81.1 | 2.01 | 6.3 | 0.60 |
| 128 | 1.6M | Muon (5 Newton-Schulz steps) | 13.0 | 3.06 | 3.05 to 3.08 | 23.6% | 81.1 | 5.03 | 6.3 | 0.60 |
| 128 | 1.6M | Adam direction, Muon step size* | 12.9 | 5.52 | 5.33 to 6.17 | 42.9% | 81.1 | 0.00 | 12.6 | 0.60 |
| 256 | 6.3M | SGD with momentum | 29.7 | 0.40 | 0.40 to 0.40 | 1.4% | 316.8 | 0.00 | 25.2 | 1.20 |
| 256 | 6.3M | AdamW | 29.7 | 0.66 | 0.66 to 0.66 | 2.2% | 316.8 | 0.00 | 50.3 | 1.23 |
| 256 | 6.3M | Signum | 29.8 | 0.69 | 0.69 to 0.69 | 2.3% | 316.8 | 0.00 | 25.2 | 1.20 |
| 256 | 6.3M | Muon, 0 Newton-Schulz steps | 29.7 | 1.10 | 1.10 to 1.10 | 3.7% | 316.8 | 0.00 | 25.2 | 1.20 |
| 256 | 6.3M | Muon, 2 Newton-Schulz steps | 29.9 | 1.87 | 1.85 to 1.87 | 6.2% | 316.8 | 16.11 | 25.2 | 1.20 |
| 256 | 6.3M | Muon (5 Newton-Schulz steps) | 29.9 | 2.83 | 2.82 to 2.84 | 9.5% | 316.8 | 40.27 | 25.2 | 1.20 |
| 256 | 6.3M | Adam direction, Muon step size* | 29.7 | 2.02 | 2.02 to 2.03 | 6.8% | 316.8 | 0.00 | 50.3 | 1.23 |
| 512 | 25.2M | SGD with momentum | 85.1 | 1.59 | 1.59 to 1.59 | 1.9% | 1252.1 | 0.00 | 100.7 | 2.50 |
| 512 | 25.2M | AdamW | 85.1 | 2.59 | 2.59 to 2.59 | 3.0% | 1252.1 | 0.00 | 201.3 | 2.60 |
| 512 | 25.2M | Signum | 84.7 | 1.96 | 1.96 to 1.96 | 2.3% | 1252.1 | 0.00 | 100.7 | 2.50 |
| 512 | 25.2M | Muon, 0 Newton-Schulz steps | 85.0 | 3.10 | 3.09 to 3.10 | 3.7% | 1252.1 | 0.00 | 100.7 | 2.50 |
| 512 | 25.2M | Muon, 2 Newton-Schulz steps | 85.1 | 6.26 | 6.23 to 6.27 | 7.4% | 1252.1 | 128.85 | 100.7 | 2.50 |
| 512 | 25.2M | Muon (5 Newton-Schulz steps) | 85.1 | 10.42 | 10.35 to 10.46 | 12.2% | 1252.1 | 322.12 | 100.7 | 2.50 |
| 512 | 25.2M | Adam direction, Muon step size* | 84.8 | 3.99 | 3.96 to 4.00 | 4.7% | 1252.1 | 0.00 | 201.3 | 2.60 |

GFLOPs are matrix products only, counted by PyTorch's FLOP counter. The element-wise work of AdamW, Signum and SGD is not counted there (it reads 0.00) and shows up in their time instead.

## Time per training step (milliseconds)

Model = forward + backward. Optimizer = the step for the attention and MLP matrices. The AdamW step for all other parameters is the same in every setting and is listed once per width.

| Optimizer | width 64: model / optimizer / overhead | width 128: model / optimizer / overhead | width 256: model / optimizer / overhead | width 512: model / optimizer / overhead |
|---|---|---|---|---|
| SGD with momentum | 9.3 / 0.30 / 3.3% | 13.0 / 0.12 / 0.9% | 29.7 / 0.40 / 1.4% | 85.1 / 1.59 / 1.9% |
| AdamW | 9.2 / 0.66 / 7.1% | 13.1 / 0.19 / 1.5% | 29.7 / 0.66 / 2.2% | 85.1 / 2.59 / 3.0% |
| Signum | 9.2 / 1.21 / 13.1% | 13.1 / 0.36 / 2.7% | 29.8 / 0.69 / 2.3% | 84.7 / 1.96 / 2.3% |
| Muon, 0 Newton-Schulz steps | 9.2 / 1.97 / 21.5% | 13.0 / 0.61 / 4.7% | 29.7 / 1.10 / 3.7% | 85.0 / 3.10 / 3.7% |
| Muon, 2 Newton-Schulz steps | 9.2 / 2.70 / 29.4% | 13.0 / 2.02 / 15.5% | 29.9 / 1.87 / 6.2% | 85.1 / 6.26 / 7.4% |
| Muon (5 Newton-Schulz steps) | 9.3 / 3.42 / 36.8% | 13.0 / 3.06 / 23.6% | 29.9 / 2.83 / 9.5% | 85.1 / 10.42 / 12.2% |
| Adam direction, Muon step size* | 12.5 / 6.07 / 48.6% | 12.9 / 5.52 / 42.9% | 29.7 / 2.02 / 6.8% | 84.8 / 3.99 / 4.7% |
| AdamW for the other parameters | 0.82 | 0.03 | 0.04 | 0.06 |

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
