# Plan: how does the optimizer shape in-context learning? (Muon vs AdamW)

The proposal is the spec; this file records how each question becomes a measurement, what
counts as a difference, and what we do if there is none. Results go in `LOG.md`.

## Setup (from the proposal)

- Tasks: w ~ N(0, I_8); prompts of K = 16 pairs, x ~ N(0, I_8), y = w.x + noise (variance 0.25).
- Pool of M tasks drawn once; M = "inf" means a fresh w for every prompt.
- Model: GPT-2-style, 8 layers, width 128, 2 heads, about 1.6M parameters. Linear read-in of
  interleaved x/y tokens and linear read-out, the format of Garg et al.
- Optimizer routing: attention (Q, K, V, O) and MLP matrices get the optimizer under study;
  everything else gets AdamW at the same settings in every run. So "AdamW vs Muon" changes only
  how the block matrices are updated.
- References on the same prompts: ridge (the Bayes-optimal answer for new tasks) and dMMSE
  (the Bayes-optimal answer if w always comes from the pool, which means pure memorisation).
- Pairing: for a given seed, both optimizers get the same task pool, initial weights and
  training batches. Differences between them are paired, which cuts noise a lot.

Deviation from the proposal: the model and training loop are written from scratch rather
than forked from Garg et al.'s repository. They keep that repository's prompt format and
model shape, but the old code pins outdated libraries, adds a curriculum, and uses GPT-2's
fused QKV matrix. A fused QKV would be orthogonalised by Muon as a single matrix. Muon is
Keller Jordan's reference update, batched over same-shaped matrices.

## What is measured at every evaluation (about 160 per run)

| Measurement | What it tells us |
|---|---|
| Error on pool tasks vs new tasks, at each context length | memorisation shows as a gap (good on pool tasks, bad on new ones) |
| Distance to ridge and to dMMSE | which algorithm the model is closest to |
| alpha, the mix weight in model ≈ alpha·dMMSE + (1−alpha)·ridge | one number from 1 (memorising) to 0 (regressing); its trajectory shows when ICL appears or fades |
| alpha at each context length | whether the model memorises with short contexts and regresses with long ones |
| Attention entropy and attention on y-tokens, per layer | whether the model reads the labels in context (ICL needs it) |
| Stable rank and effective rank of every block matrix, and of its change since init | the "how": Muon's orthogonalised updates should spread learning across more directions |

Checkpoints are saved at 10 log-spaced steps, so we can add linear probes later without retraining.

## Primary results (fixed before looking)

1. **Task-diversity threshold M\***: the pool size where alpha on new tasks crosses 0.5
   (closer to ridge than to dMMSE), from a fit across the M grid. The difference is
   log2 M\*(Muon) − log2 M\*(AdamW), with a bootstrap CI over seeds.
2. **ICL onset time**: the first step where error on new tasks at the last position falls
   halfway from the no-context error (8.25) to ridge error. Paired across seeds.
3. **Transience**: at pool sizes near M\*, in long runs, the step where error on new tasks
   rises 10% above its minimum (if it does at all).
4. **Examples needed**: the smallest context length k where error is within 10% of ridge at
   the same k, and the area between the model's error-vs-k curve and ridge's.

Significance: paired sign-flip permutation test across seeds (6 seeds gives a minimum two-sided
p of 0.03), with effect sizes and CIs. A null result gets an equivalence bound ("any
difference in log2 M\* is smaller than X"), which is also reportable.

## The mechanism hypothesis we are testing

Muon equalises the singular values of each update, so it pushes many directions at once
instead of the few dominant ones Adam favours. Wang et al. 2025 (arXiv:2509.26030) report that
Muon learns rare ("tail") associations better than Adam, and Li et al. 2026 (arXiv:2602.05725)
model this in associative memories. Muon also reaches grokking faster than AdamW on modular
arithmetic (arXiv:2607.20512), which is another switch from memorising to generalising. An
arXiv search on 2026-11-03 found no study of Muon and the task-diversity threshold for ICL.
If the tail-memory result carries over here, **Muon memorises larger task pools,
so its M\* is higher**: it needs more task diversity before switching to ICL. It may also
memorise faster (earlier, stronger transient memorisation) and keep higher-rank weight changes.
The opposite outcome is possible: ICL relies on a low-rank attention circuit, and a uniform
spectrum could speed up finding it. Either way, the spectral measurements show which.

## Run sequence

| Run | What | Output you will see | Decision it drives |
|---|---|---|---|
| 1 Calibration | M=inf LR sweep (4 per optimizer) + coarse M scan (8, 128, 2048, 32768), 30k steps, 1 seed | Best LR per optimizer; whether ICL appears in 30k steps; roughly where M\* sits; runtime per run | LRs and training length for Run 2; M grid |
| 2 Main sweep | M grid of about 10 sizes around M\*, both optimizers, 6 seeds | M\* per optimizer with CI; alpha-vs-M curves; onset times | Is there a significant difference? |
| 3 Dynamics | 3 pool sizes near M\*, long training (several times Run 2), both optimizers, 6 seeds | Whether ICL fades (transience) and when, per optimizer | Question 2 of the proposal |
| 4 Mechanism | Spectra/attention from Runs 2-3, plus probes on saved checkpoints | Why they differ | The explanatory part of the write-up |
| 5 Robustness | Key cells at ±1 LR step and with weight decay | Whether the difference survives tuning | Whether we can claim it |

## Fallbacks if Muon and AdamW look the same

In order of cost. Each is a real result, not just a rescue.

1. **Report the null rigorously**: equivalence bounds on M\* and onset. "Swapping the optimizer
   does not move the memorisation-to-ICL threshold by more than X" is a useful finding.
2. **Isolate the ingredient**: Signum (sign of momentum: normalised but not orthogonalised) and
   SGD with momentum (not normalised) are already implemented. If AdamW≈Muon but SGD differs,
   the result becomes "update normalisation matters, orthogonalisation does not".
3. **Regimes where optimizers matter more**: weight decay (known to control transience), a
   smaller model (memorisation competes harder for capacity), larger d (16 or 32), longer
   training, and a Zipfian task distribution instead of uniform.
4. **A different task where the ICL vs memorisation competition is sharper**: in-context
   classification with Gaussian-mixture labels (Reddy 2023; Chan et al. 2022), where
   transient ICL is robust and in-weights vs in-context learning compete directly.
