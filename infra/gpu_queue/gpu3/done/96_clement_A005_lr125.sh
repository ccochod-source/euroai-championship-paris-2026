#!/bin/bash
# TIMEOUT=3600
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name clem_A005_lr125 --A 0.05 --alpha 4 --lr 1.25e-4 --steps 4000
