#!/bin/bash
# TIMEOUT=3600
set -euo pipefail
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
/p/scratch/training2666/cochod2/euroai-paris/unlearning/.venv/bin/python -u /p/scratch/training2666/cochod2/euroai-paris/unlearning/codex/oof_train.py --name oof_calibrated_s14 --strength 1.4 --targets oof_targets.pt --baseline orehn/ckpt/ft6_A03r5_g4_last6_t1.pt
