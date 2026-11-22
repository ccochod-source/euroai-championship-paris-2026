#!/bin/bash
# TIMEOUT=2400
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/finetune.py --name U005_ho --A 0.05 --alpha 0.5 --lr 1e-4 --steps 3000 --holdout 0 --save 0
