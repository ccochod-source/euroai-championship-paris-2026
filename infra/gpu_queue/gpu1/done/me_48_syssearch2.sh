#!/bin/bash
# TIMEOUT=9000
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_DATE="21 Nov 2026"
python sys_search.py --model Qwen/Qwen2.5-3B-Instruct --ep ep-02 --out out/sys_search/ep-02_b.json --pool-file sys_pool2.txt --append-only --start "Help the user with their questions and tasks." --rounds 6
