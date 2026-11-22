#!/bin/bash
# TIMEOUT=1200
set -euo pipefail
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
python -u codex/train.py --mode npo --name npo_stable_cap3_last4_s900 --steps 900 --beta 0.5 --cap 3 --last 4 --gamma 4 --alt-kl 1 --recovery 1 --ref-conf 0.8 --extra-bright 0.35 --full-eval
