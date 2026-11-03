#!/bin/bash
# Tiny local run of every optimizer: checks the code path, not the science.
# usage: scripts/smoke.sh <outdir>
set -euo pipefail
out=${1:?outdir}
cd "$(dirname "$0")/.."
for spec in adamw:1e-3 muon:0.02 signum:1e-3 sgdm:0.05; do
  o=${spec%%:*}; lr=${spec##*:}
  python3 train.py --opt "$o" --lr "$lr" --M 16 --steps 300 --n_layer 2 --n_embd 64 --warmup 50 \
    --out "$out" --device cpu 2>&1 | tail -1
done
python3 scripts/smoke_summary.py "$out"
