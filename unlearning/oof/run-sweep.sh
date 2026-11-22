#!/bin/bash
# usage: run-sweep.sh OUTDIR depth:lr:tau [...]
set -euo pipefail
umask 002
root=/p/scratch/training2666/cochod2/euroai-paris/unlearning
out=$1; shift
mkdir -p "$out"
cd "$root"
export PYTHONPATH="$root"
for config in "$@"; do
    depth=$(echo "$config" | cut -d: -f1); lr=$(echo "$config" | cut -d: -f2); tau=$(echo "$config" | cut -d: -f3)
    name="oof-d$depth-lr$lr-t$tau"
    .venv/bin/python -u oof/oof_finetune.py --teacher oof/oof-wd0.01.pt --depth "$depth" --lr "$lr" --tau "$tau" \
        --steps 300 --every 100 --out "$out/$name.pt" 2>&1 | tee "$out/$name.log"
done

