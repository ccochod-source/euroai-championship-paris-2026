#!/bin/bash
# TIMEOUT=3600
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name clem_A005_lr15 --A 0.05 --alpha 4 --lr 1.5e-4 --steps 4000
