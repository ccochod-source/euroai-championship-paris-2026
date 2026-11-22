#!/bin/bash
# TIMEOUT=3000
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_DATE="21 Nov 2026"
python sys_search.py --model unsloth/Llama-3.2-1B-Instruct --ep ep-29 --out out/sys_search/ep-29.json
python sys_search.py --model Qwen/Qwen2.5-1.5B-Instruct --ep ep-08 --out out/sys_search/ep-08.json
