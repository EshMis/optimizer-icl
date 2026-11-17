# Why does Muon memorise more? Mechanism, capacity and a second task

This document continues [WRITEUP.md](WRITEUP.md). That write-up established *what* differs
between Muon and AdamW on in-context linear regression: at a matched number of steps Muon
needs about twice the task diversity (M* of 80 against 41) before in-context learning wins,
and it left open *why*. Runs 6 to 8 answer that. Every number here is produced by a script
under `scripts/` from saved metrics; see [Where each number comes from](#where-each-number-comes-from).

## The short version

- **It is not a clock effect.** Giving AdamW's update Muon's step length moves the
  threshold by a quarter of the gap. Re-plotting the threshold against distance travelled in
  weight space, distance from initialisation, or training loss never brings the optimizers
  together. Muon changes what is learned, not only how fast.
- **The cause is orthogonalisation, applied fully.** Muon with its orthogonalisation
  switched off (same momentum, same step length) sits exactly on AdamW. Two Newton-Schulz
  iterations, which only partly flatten the update spectrum, barely move the threshold.
  Five iterations reproduce Muon.
- **The memorising happens in the MLP.** Muon applied to the MLP matrices alone gives
  about 80% of the gap. Muon applied to the attention matrices alone gives none of it.
  The effective rank of the MLP weight change separates every memorising setting from
  every non-memorising one, whatever the attention matrices do.
- **The gap is a constant factor of about 2 at every model width** from 64 to 512 (0.4M to
  25M parameters). Both thresholds grow with width at the same rate, so Muon does not scale
  differently, it uses a fixed larger share of the same capacity.
- **Signum sits between AdamW and Muon even at matched step length**, which reconciles the
  original Signum result: equalising entries recovers a quarter of the gap, equalising singular
  directions recovers all of it.
- **On a second task, letter-substitution ciphers,** Muon again memorises larger pools, and it
  also finds the in-context copy circuit far faster than AdamW, which often stays stuck on a
  plateau. That plateau makes the cipher threshold comparison unreliable so far; a calibration
  run (8b) is in progress.

## Run 6: which ingredient of Muon's update, and in which block

### Design

Pool sizes 16, 32, 64, 128 and 256, three paired seeds, 30,000 steps, the model of the main
runs (8 layers, width 128). Eight settings, each a single change to the update of the block
weight matrices; everything else is AdamW at 3e-4 as before.

| Setting | What it changes |
|---|---|
| AdamW 3e-4 | the baseline |
| Adam direction, Muon step size | Adam's update, rescaled per matrix to the Frobenius norm a Muon update has |
| Muon, 0 Newton-Schulz steps | Muon's momentum, normalised to Muon's step length, no orthogonalisation |
| Muon, 2 Newton-Schulz steps | partial orthogonalisation at the same step length |
| Muon, 5 Newton-Schulz steps, norm-matched | full orthogonalisation at exactly the same step length (a control for the knob) |
| Muon 0.005 | the baseline |
| Muon on attention only | Q, K, V, O get Muon; the MLP matrices get AdamW 3e-4 |
| Muon on MLP only | the reverse |

New measurements: the cumulative Frobenius length of the path each block's weights travel,
and the Frobenius norm of each matrix's change since initialisation (alongside the effective
rank already measured).

### Result

![alpha against pool size, all eight settings](reports/figs/run06/alpha_vs_M.png)

| Setting | M* (tasks) | log2 M* per seed | vs AdamW (doublings, per seed) |
|---|---|---|---|
| AdamW 3e-4 | 41 | 5.43 5.21 5.49 | 0 |
| Adam direction, Muon step size | 48 | 5.51 5.54 5.69 | +0.21 (+0.08 +0.33 +0.20) |
| Muon, 0 NS steps | 41 | 5.39 5.29 5.34 | -0.03 (-0.04 +0.08 -0.14) |
| Muon, 2 NS steps | 45 | 5.50 5.40 5.58 | +0.12 (+0.07 +0.19 +0.09) |
| Muon, 5 NS steps, norm-matched | 77 | 6.23 6.15 6.42 | +0.89 (+0.80 +0.95 +0.93) |
| Muon 0.005 | 80 | 6.35 6.31 6.31 | +0.95 (+0.92 +1.10 +0.83) |
| Muon on attention only | 36 | 5.27 4.90 5.29 | -0.22 (-0.16 -0.31 -0.19) |
| Muon on MLP only | 71 | 6.16 5.92 6.39 | +0.78 (+0.73 +0.72 +0.90) |

The settings fall into two groups with nothing in between: {AdamW, Adam direction at Muon's
step, 0 and 2 Newton-Schulz steps, Muon on attention only} at 36 to 48 tasks, and {Muon, 5
Newton-Schulz steps norm-matched, Muon on MLP only} at 71 to 80 tasks. Every difference
between the groups has the same sign in 3 of 3 seeds.

**Step size is not the cause.** The Adam-direction control walks 7 times further in weight
space than AdamW (path length 53,000 against 7,400 for the attention matrices at 64 tasks)
and ends 5 times further from initialisation, yet its threshold is within a quarter of a
doubling of AdamW's. The 0-step Muon variant has the same step length as Muon and the same
threshold as AdamW.

**Orthogonalisation is the cause, and it has to be complete.** With the step length held
fixed, 0 and 2 Newton-Schulz iterations leave the threshold at AdamW's, and 5 iterations
reproduce Muon's. Two iterations of the quintic Newton-Schulz polynomial leave the small
singular values of the update small; five bring every singular value near 1. So it is the
flat spectrum itself, not partial whitening, that memorises.

**The MLP does the memorising.** Routing Muon to the MLP alone recovers 0.78 of the 0.95
doublings; routing it to attention alone recovers none and lands slightly below AdamW. The
effective rank of the MLP weight change since initialisation, at 64 tasks:

| Setting | eff. rank, attention | eff. rank, MLP | alpha on new tasks |
|---|---|---|---|
| AdamW 3e-4 | 53 | 85 | 0.36 |
| Adam direction, Muon step size | 54 | 80 | 0.36 |
| Muon, 0 NS steps | 32 | 56 | 0.35 |
| Muon, 2 NS steps | 35 | 74 | 0.36 |
| Muon on attention only | 78 | 70 | 0.37 |
| Muon on MLP only | 44 | 115 | 0.54 |
| Muon, 5 NS steps, norm-matched | 76 | 114 | 0.57 |
| Muon 0.005 | 77 | 114 | 0.58 |

A high-rank attention change (attention-only Muon, 78) does nothing; a high-rank MLP change
(114 to 115) is present in exactly the settings that memorise more. (Seed 0; the other seeds
agree.)

**No alternative clock collapses the curves.**

![threshold over training against four clocks](reports/figs/run06/threshold_vs_clocks.png)

The threshold M*(t) is computed at every evaluation and plotted against the training step,
the cumulative weight-path length, the distance from initialisation and the training loss.
On every axis the settings split into the same two groups. The original write-up's "Muon is
AdamW on a 4x faster clock" describes the step axis well, but there is no quantity along
which the two optimizers follow one curve. The difference is in the solution reached.

### Interpretation

Memorising a pool of tasks means storing many independent associations in the MLP weights.
Adam-style updates concentrate each step in a few dominant singular directions, so stored
associations compete for the same subspace and memorisation saturates early. An
orthogonalised update spreads each step evenly over every direction, so the MLP's full
capacity is available for storing tasks, and the model needs about twice the task diversity
before storing them stops paying off. This is the behaviour predicted for spectral optimizers
by the associative-memory analyses of Wang et al. 2025 and Li et al. 2026, and the ablation
reaches the same "orthogonalisation is the active ingredient" conclusion as the grokking
study of arXiv:2607.20512, with the sign reversed: there the ingredient brings generalisation
sooner, here it brings memorisation sooner.

It also explains the loose end of the main write-up. Muon's threshold did not move with its
learning rate because the learning rate sets how far each step goes, and that does not
matter; the shape of the update spectrum does, and it is fixed.

## Run 7: capacity (width) and the Signum discrepancy

### Design

(a) Model width 64, 256 and 512 for AdamW 3e-4 and Muon 0.005 (width 128 from Run 6), pools 16
to 256, 2 seeds (3 at width 128), 30,000 steps. Depth, heads and everything else unchanged.
The hypothesis: if Muon memorises more because it uses the MLP's storage capacity more fully,
its threshold should grow with width faster than AdamW's. (b) Signum with its update rescaled
to Muon's Frobenius norm, 3 seeds, to separate the sign operation from the step length.

### Result (a): the gap is a constant factor at every width

![threshold against width](reports/figs/run07/threshold_vs_width.png)

| Width | Parameters | AdamW M* | Muon M* | Muon minus AdamW (doublings) |
|---|---|---|---|---|
| 64 | 0.40M | 25 | 50 | +0.98 |
| 128 | 1.59M | 41 | 80 | +0.95 |
| 256 | 6.33M | 57 | 100 | +0.81 |
| 512 | 25.2M | 64 | 124 | +0.94 |

Both thresholds grow with width, and at the same rate: M* grows roughly as width^0.45 for
AdamW and width^0.43 for Muon. Muon's threshold is about twice AdamW's at every width, and
higher in every seed at every width.

So the hypothesis of a different scaling exponent is not supported. Width raises how many tasks
both optimizers can memorise; Muon's advantage is a fixed factor of about 2 on top of that,
not a steeper growth. In the bookshelf picture: a bigger bookshelf holds more books for both
students, and Muon always fits about twice as many on a shelf of a given size. With 2 seeds
per width this is a measurement of the trend, not a precise exponent.

### Result (b): Signum sits between, at matched step length

| Setting (all at Muon's Frobenius norm except the baselines) | M* | log2 M* per seed |
|---|---|---|
| AdamW 3e-4 | 41 | 5.43 5.21 5.49 |
| Normalised momentum (Muon, 0 NS steps) | 41 | 5.39 5.29 5.34 |
| Signum, norm-matched | 51 | 5.67 5.55 5.83 |
| Muon 0.005 | 80 | 6.35 6.31 6.31 |

This reconciles WRITEUP.md, which found Signum between AdamW and Muon, with Run 6, which found
normalised momentum on AdamW. At the same step length, rescaling the whole matrix does
nothing, equalising the size of every entry (sign) recovers about a quarter of the gap (0.31
doublings, 3 of 3 seeds above both AdamW and normalised momentum), and equalising every
singular direction (orthogonalisation) recovers all of it.

One caveat: the effective rank of Signum's MLP weight change (79 at 64 tasks, seed 0) is no
higher than AdamW's (85), so the effective-rank measure does not explain Signum's partial
effect. Effective rank tracks the full-Muon settings well; what Signum does is not captured
by it.

## Run 8: a second task, letter-substitution ciphers

### The task

An alphabet of 16 letters and a hidden cipher, a permutation that maps each plain letter to
a coded letter. A prompt is 24 (plain, coded) pairs with plain letters drawn at random; the
model predicts each coded letter from the pairs before it. Two cases arise. If the plain letter
appeared earlier in the prompt, the answer can be copied from the context. If it is new, the
best possible guess is uniform over the coded letters not yet used. A pool of M ciphers is
fixed before training, exactly as the pool of weight vectors is in regression.

Two exact reference predictors play the roles of ridge and dMMSE: the Bayes-optimal in-context
decoder (uniform prior over all permutations) and the Bayes-optimal memoriser (uniform prior
over the pool). Both were checked against brute-force enumeration and agree exactly. alpha is
defined on the predicted probability vectors exactly as before.

The model is the same transformer with a token embedding in place of the linear read-in and a
16-way softmax in place of the scalar read-out. Settings: AdamW 3e-4, Muon 0.005 and Muon on
the MLP only, pools 16 to 4,096 plus unlimited, 2 seeds, 15,000 steps.

### What happened: a training plateau, and Muon escapes it reliably

Learning to decode in context requires an attention circuit that finds the earlier occurrence
of the same plain letter and copies its code, the "induction head" pattern known from language
models. Before that circuit forms, training sits on a plateau with loss near 2.3, the loss of
guessing among unused codes without ever copying.

Full Muon escaped the plateau in every run, at every pool size, between steps 600 and 1,100.
AdamW and Muon-on-MLP, whose attention matrices are both trained by AdamW at 3e-4, often did
not: at pool sizes of 1,024 and above, and with an unlimited pool, 6 of 8 AdamW runs and 5 of 8
Muon-on-MLP runs were still on or near the plateau at step 15,000. Those runs neither memorise nor decode, so their alpha is
meaningless, and the threshold computed from them is not a valid reading. The threshold table in
`reports/run08_table.md` is therefore kept for the record but not interpreted.

Two things can be read from this run already:

- **Where both memorise, Muon memorises more again.** At 256 ciphers, every Muon run fits its
  pool perfectly (accuracy on repeated letters 1.00 on pool ciphers, alpha 0.57 on new ciphers),
  while AdamW is at alpha 0.36 with one of its two seeds failing to fit at all.
- **Muon finds the in-context copy circuit much faster.** On the unlimited pool, full Muon
  started copying by step 1,110 and ended within 0.01 of the optimal decoder's cross-entropy in
  seed 0 (seed 1 started by step 981 and ended 0.16 above optimal), while both AdamW seeds were
  still on the plateau at step 15,000. Muon on
  the MLP only behaved like AdamW, which places this speed-up in the attention matrices, the
  opposite block from the memorisation effect.

That second point is new relative to the regression task, where both optimizers learned
in-context regression easily. On the cipher task Muon's orthogonalised updates help attention
form the copy circuit, while the same updates help the MLP memorise.

### Run 8b: calibration (running)

To get a valid threshold comparison, Run 8b trains on the unlimited pool for 30,000 steps with
3 seeds, at AdamW 3e-4, 1e-3 and 3e-3, Muon 0.005 and Muon-on-MLP, to find settings under which
every seed escapes the plateau. The pool-size sweep will then be rerun at those settings.

## Where each number comes from

| Section | Report | Command |
|---|---|---|
| Run 6 | `reports/run06_table.md`, `reports/figs/run06/` | `python scripts/run06_mech.py results/run06_mech reports/figs/run06` |
| Run 7 | `reports/run07_table.md`, `reports/figs/run07/` | `python scripts/run07_capacity.py results/run06_mech results/run07a_width results/run07b_signum reports/figs/run07` |
| Run 8 | `reports/run08_table.md`, `reports/figs/run08/` | `python scripts/run08_cipher.py results/run08_cipher reports/figs/run08` |
| Run 8, per-run plateau check | printed table | see the inline check in this section; criterion: accuracy on repeated letters of pool ciphers at least 0.9 |

Run definitions are in `tasks/run06_mech_g*.txt`, `tasks/run07a_width.txt`,
`tasks/run07b_signum_fm.txt` and `tasks/run08_cipher.txt`. New `train.py` flags: `--ns_steps`,
`--fro_match`, `--route {all,attn,mlp}`, `--eval_pct`, and `--opt adamw_fn`. Runs were on
RTX A6000 GPUs, five or six runs sharing a GPU, 32-bit floats.
