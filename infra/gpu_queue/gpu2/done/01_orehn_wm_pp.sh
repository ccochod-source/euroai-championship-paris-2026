#!/bin/bash
# TIMEOUT=1200
umask 002
W=/p/scratch/training2666/dasilvalopes3; PY=$W/wm/kit/.venv_team/bin/python
cd /p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task/orehn || exit 1
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=16
$PY -u assemble3.py
