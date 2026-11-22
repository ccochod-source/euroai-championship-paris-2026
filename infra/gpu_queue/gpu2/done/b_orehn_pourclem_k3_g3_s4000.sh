#!/bin/bash
# TIMEOUT=900
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name pc_oof_k3_g3_last6_s4000 --targets orehn/cache/q_oof_k3.pt --gamma 3 --train last6 --lr 1e-4 --steps 4000
