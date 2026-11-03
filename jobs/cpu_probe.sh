#!/bin/bash
#$ -S /bin/bash
# usage: qsub -terse -pe def_slot 16 -l mem_req=2G,s_vmem=2G,h_rt=00:30:00 -N oicl_cpu -o logs/ -e logs/ code/jobs/cpu_probe.sh
ROOT=${OICL_ROOT:-$HOME/optimizer-icl}
cd "$ROOT" || exit 2
# shellcheck source=/dev/null
[[ -f code/jobs/site.env ]] && . code/jobs/site.env
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
echo "host=$(hostname)"; lscpu | grep -E "Model name"
"${CPU_PY:-python3}" code/scripts/time_steps.py cpu 4+8+16
