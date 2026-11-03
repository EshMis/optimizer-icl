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
| Stage | Run 1 (calibration) running on Shirokane CPU nodes since 2026-11-03 10:20 PT, results expected about 11:45 PT |
| Known so far | Code runs end to end with all four optimizers (local smoke test, not a result) |
| Next decision | Learning rates and training length for the main sweep |

---

## Run 1: calibration (started 2026-11-03 10:20 PT)

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

**Result:** pending.
