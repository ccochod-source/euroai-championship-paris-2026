#!/bin/bash
# TIMEOUT=3000
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name ft2_A015 --A 0.15 --alpha 4 --lr 1e-4 --steps 4000
