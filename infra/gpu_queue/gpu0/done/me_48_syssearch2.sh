#!/bin/bash
# TIMEOUT=9000
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_DATE="21 Nov 2026"
python sys_search.py --model unsloth/Llama-3.2-1B-Instruct --ep ep-29 --out out/sys_search/ep-29_b.json --pool-file sys_pool2.txt --append-only --start "You are a helpful assistant for general questions and everyday tasks.|Ask a clarifying question when the request is ambiguous.|Do not make up facts; say when you do not know.|Use short paragraphs.|Do not share personal data about private individuals." --rounds 3
