#!/bin/bash
# TIMEOUT=3600
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name clem_A03r4_g4_last6 --A 0.3 --alpha 4 --rpow 4 --gamma 4 --train last6 --lr 1e-4 --steps 3000
