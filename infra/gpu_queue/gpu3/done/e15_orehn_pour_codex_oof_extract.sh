#!/bin/bash
# TIMEOUT=3600
set -euo pipefail
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -u codex/oof_extract.py
