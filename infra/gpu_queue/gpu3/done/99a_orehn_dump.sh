#!/bin/bash
# TIMEOUT=1500
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/dump.py /p/scratch/training2666/cochod2/euroai-paris/unlearning/output/oof-15902873/oof-d6-lr3e-5-s300.pt
