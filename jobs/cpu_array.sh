#!/bin/bash
#$ -S /bin/bash
# One training run per array task, on CPU. Task i = i-th non-comment line of $TASKS.
# usage (from anywhere):
#   qsub -terse -t 1-<n> -pe def_slot 16 -l mem_req=2G,s_vmem=2G,h_rt=04:00:00,hostname='!hc002' \
#        -N oicl_<run> -o $HOME/optimizer-icl/logs/ -e $HOME/optimizer-icl/logs/ \
#        -v TASKS=tasks/<run>.txt $HOME/optimizer-icl/code/jobs/cpu_array.sh
# Machine-specific paths (CPU_PY, GPU_PY) come from jobs/site.env, which is not committed.
ROOT=${OICL_ROOT:-$HOME/optimizer-icl}
# shellcheck source=/dev/null
[[ -f $ROOT/code/jobs/site.env ]] && . "$ROOT/code/jobs/site.env"
PY=${CPU_PY:-python3}
cd "$ROOT" || exit 2
: "${TASKS:?set TASKS}" "${SGE_TASK_ID:?array job only}"
n=${NSLOTS:-1}
export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n
line=$(grep -v '^#' "$TASKS" | grep . | sed -n "${SGE_TASK_ID}p")
[[ -n "$line" ]] || { echo "no line $SGE_TASK_ID in $TASKS" >&2; exit 2; }
name=$(basename "$TASKS" .txt)
echo "host=$(hostname) slots=$n task=$SGE_TASK_ID: $line"
# shellcheck disable=SC2086
exec $PY code/train.py $line --out "$ROOT/results/$name" --device cpu
