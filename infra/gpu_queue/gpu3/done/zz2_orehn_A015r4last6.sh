#!/bin/bash
# TIMEOUT=2400
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name ft7_A015r4_g4_last6_t1 --A 0.15 --alpha 4 --rpow 4 --gamma 4 --t1 0.5 --train last6 --lr 1e-4 --steps 3000
