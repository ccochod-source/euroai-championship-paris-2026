#!/bin/bash
# TIMEOUT=2400
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name pc_oof_k4_g8_last6 --targets orehn/cache/q_oof_k4.pt --gamma 8 --train last6 --lr 1e-4 --steps 3000
