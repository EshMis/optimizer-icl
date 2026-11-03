#!/bin/bash
#$ -S /bin/bash
# One H100, several training runs sharing it.
# usage (from $HOME/optimizer-icl):
#   qsub -terse -q gpuh.q -pe def_slot 8 -l h100=1,mem_req=48G,s_vmem=48G,h_rt=08:00:00 \
#        -N oicl_<run> -o $HOME/optimizer-icl/logs/ -e $HOME/optimizer-icl/logs/ -v TASKS=tasks/<run>.txt,PAR=8 $HOME/optimizer-icl/code/jobs/pack.sh
set -uo pipefail
ROOT=${OICL_ROOT:-$HOME/optimizer-icl}
# shellcheck source=/dev/null
[[ -f $ROOT/code/jobs/site.env ]] && . "$ROOT/code/jobs/site.env"
PY=${GPU_PY:-python3}
cd "$ROOT" || exit 2
: "${TASKS:?set TASKS}" "${PAR:?set PAR}"
assigned="${SGE_HGR_h100:-}"; assigned="${assigned// /}"
[[ -n "$assigned" ]] || { echo "no scheduler-assigned H100" >&2; exit 2; }
for cand in "$assigned" 0 unset; do
  if [[ $cand == unset ]]; then unset CUDA_VISIBLE_DEVICES; else export CUDA_VISIBLE_DEVICES=$cand; fi
  n=$($PY -c "import torch;print(torch.cuda.device_count())" 2>/dev/null)
  [[ "$n" == 1 ]] && break
done
[[ "$n" == 1 ]] || { echo "torch sees $n GPUs under every binding" >&2; nvidia-smi -L >&2; exit 3; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset} tasks=$TASKS par=$PAR"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
name=$(basename "$TASKS" .txt)
$PY code/sweep.py "$TASKS" --parallel "$PAR" --out "$ROOT/results/$name" --logdir "$ROOT/logs/$name"
