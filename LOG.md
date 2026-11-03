# Research log: Muon vs AdamW and in-context learning

Newest entry on top. Each run gets one entry: what ran, what came out, what it means, and
what happens next. Engineering detail (job IDs, fixes, cluster issues) is kept out of this
file and lives in `../optimizer-icl-home/memory/STATUS.md`. The design and the fallback ladder are in `PLAN.md`.

How to read the numbers:
- **Error on new tasks** is the mean squared error on prompts whose w was never in training,
  after 15 examples. It is shown next to its ratio to ridge regression, the best possible
  answer: "1.00x" means the model is doing optimal in-context regression.
- **Error on pool tasks** is the same measure on prompts whose w came from the training
  pool. It is compared with dMMSE, the best possible answer for a model that memorises the pool.
- **alpha** places the model between the two: about 1 means it behaves like a pool memoriser,
  about 0 means it does real regression. It only means something once the model is close
  to one of them, so ignore it in the first few hundred steps.
- **ICL onset** is the step where new-task error has fallen halfway from where it started
  to ridge.

## Where we are

| | |
|---|---|
| Stage | Run 2 (main pool-size sweep, 240 runs) running since 2026-11-03 12:33 PT; seeds 0-3 expected about 16:30 PT, seeds 4-5 after |
| Known so far | Run 1, one seed: both optimizers reach the same quality on new tasks once tuned. At pool size 128 ICL appears and then fades in both, and Muon goes through both phases about 4x sooner. |
| Next decision | Whether that early-rise, early-fade pattern holds across seeds and learning rates (Run 2) |

---

## Run 2: main pool-size sweep (started 2026-11-03 12:33 PT)

**What:** 240 runs of 30,000 steps on Shirokane CPU nodes: pool sizes M = 16, 32, 64, 128, 256,
512, 1024, 2048, 4096 and inf; 6 seeds; two learning rates per optimizer, AdamW at 1e-4 and 3e-4
and Muon at 0.005 and 0.01. These are the best two of each from Run 1. The non-matrix
parameters use AdamW at 3e-4 in every run, so the only thing that changes is how the attention
and MLP matrices are trained. Seed s gives both optimizers the same pool, starting weights and
batches, so every comparison is paired.

**What the results will look like:**
- A curve of alpha (memorising vs regressing) against pool size for each optimizer, with error
  bars over seeds. The pool size where it crosses 0.5 is M*, the task-diversity threshold.
- A stats table: for each pool size, the Muon-minus-AdamW difference in new-task error,
  pool-task error, onset step and fade, with a 95% interval and a paired p-value. The seed is
  the independent unit, so 6 seeds (4 in the first batch, 2 more queued behind them) give a
  smallest possible p of 0.031.
- Mechanism figures: effective rank of each weight change, and how much attention goes to
  the y values, over training.

**What would count as a finding:** the same sign of difference in every seed at both learning
rates, for at least one of: M*, onset step, or fade. If Run 1's pattern is real, Muon should
peak and fade earlier at M = 64 to 256, and the error-vs-M curves should cross.

**Result:** pending.

---

## Run 1: calibration (2026-11-03, 10:20-12:30 PT)

**What:** 17 training runs of 30,000 steps each, seed 0, run side by side on Shirokane CPU nodes
(every H100 on the cluster was taken; a CPU node does one run in about an hour).
- 9 runs at M = inf (a fresh task every prompt) to pick learning rates: AdamW at 1e-4, 3e-4,
  1e-3, 3e-3; Muon at 0.005, 0.01, 0.02, 0.04 (non-matrix parameters on AdamW 1e-3), plus
  Muon 0.02 with AdamW 3e-4 on the rest.
- 8 runs scanning pool size M = 8, 128, 2048, 32768 for both optimizers at default learning
  rates, to locate roughly where the switch from memorising to ICL happens.

**What the results will look like:** a table with one row per run (the columns above), and
three figures in `reports/figs/run01/`: new-task error over training, pool-task error over
training, and alpha over training, one panel per M.

**What we expect:** at M = inf both optimizers should approach ridge (about 1.0-1.3x). At
M = 8 both should memorise (pool-task error near dMMSE, alpha near 1, poor on new tasks). At
M = 32768 both should do ICL. The interesting cells are 128 and 2048. Seed 0 alone cannot show
a significant difference; this run only sets up the main sweep.

**Result:** all 17 runs finished. Full table: `reports/run01_table.md`. Figures: `reports/figs/run01/`.

Learning rates, with a fresh task in every prompt (M = inf):

| Optimizer | Learning rate | New-task error at 15 examples (x ridge) | Onset step |
|---|---|---|---|
| AdamW | 1e-4 | 0.581 (1.08x) | 880 |
| AdamW | 3e-4 | 0.591 (1.09x) | 586 |
| AdamW | 1e-3 | 0.637 (1.18x) | 511 |
| AdamW | 3e-3 | did not learn | never |
| Muon | 0.005 | 0.579 (1.07x) | 446 |
| Muon | 0.01 | 0.580 (1.07x) | 340 |
| Muon | 0.02 | 0.593 (1.10x) | 297 |
| Muon | 0.04 | 0.658 (1.22x) | 226 |

Pool-size scan (AdamW 1e-3, Muon 0.02; neither is its best rate):

| M | AdamW new-task error | Muon new-task error | Best new-task error, AdamW / Muon (step) | Fade (final / best), AdamW / Muon |
|---|---|---|---|---|
| 8 | 8.77 (memorises; pool error 1.06x dMMSE) | 10.53 (pool error 1.71x dMMSE, unstable) | no ICL | no ICL |
| 128 | 1.80 | 2.38 | 0.93 (7,200) / 1.01 (1,800) | 1.99 / 2.29 |
| 2048 | 0.603 | 0.624 | still improving | none |
| 32768 | 0.604 | 0.603 | still improving | none |

**What it means:**
1. Tuned, the two optimizers end up equally good at in-context regression: about 1.07x the
   optimal error after 30,000 steps. So any difference is in *how* they get there and *when*
   they switch between memorising and regressing, not in what they can eventually reach.
2. Muon starts doing ICL sooner at every learning rate we can compare. This depends heavily on
   learning rate, though: AdamW at 1e-3 starts sooner than Muon at 0.005. The fair comparison is
   between tuned settings, and Run 2 does that across seeds.
3. **Transient ICL appeared on its own at M = 128.** Both models first learn to regress from the
   prompt (new-task error drops to about 1.0), then slowly swap that for memorising the 128
   tasks: new-task error doubles while pool-task error keeps falling. This is the
   rise-then-fade the proposal asked about (Singh et al. 2023). Muon reached its ICL peak about
   4x sooner (step 1,800 vs 7,200) and had faded further by the end. That is one seed at
   untuned learning rates, but if it holds it answers the proposal's question 2.
4. At M = 8, Muon at 0.02 did not settle: its pool error kept spiking. That learning rate is
   too high when there are few tasks, which is another reason Run 2 uses lower ones.
5. With 2,048 or more tasks, both do ICL and look the same at this training length. So the
   threshold M* is between 128 and 2,048 here. Run 2 fills that range in with powers of two.

**Deviation:** the job ran on CPU nodes instead of the planned H100 because every H100 was busy.
Every run in a given comparison uses the same hardware.
