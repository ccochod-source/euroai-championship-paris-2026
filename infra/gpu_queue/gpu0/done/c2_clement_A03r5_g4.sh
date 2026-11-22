#!/bin/bash
# TIMEOUT=3600
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name clem_A03r5_g4 --A 0.3 --alpha 4 --rpow 5 --gamma 4 --lr 1e-4 --steps 3000
