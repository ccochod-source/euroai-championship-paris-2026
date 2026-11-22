#!/bin/bash
# TIMEOUT=2400
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name ft6_A03r5_g4_last6_t1 --A 0.3 --alpha 4 --rpow 5 --gamma 4 --t1 1.0 --train last6 --lr 1e-4 --steps 3000
