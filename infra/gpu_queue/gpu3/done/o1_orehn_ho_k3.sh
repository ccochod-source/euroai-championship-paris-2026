#!/bin/bash
# TIMEOUT=2400
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name ho_oof_k3_g4_last6 --targets orehn/cache/q_oof_k3.pt --gamma 4 --train last6 --lr 1e-4 --steps 3000 --holdout 0 --save 0
