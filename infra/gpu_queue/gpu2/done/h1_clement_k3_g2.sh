#!/bin/bash
# TIMEOUT=3600
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name clem_oof_k3_g2_last6 --targets orehn/cache/q_oof_k3.pt --gamma 2 --train last6 --lr 1e-4 --steps 3000
