#!/bin/bash
# TIMEOUT=3600
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_DATE="21 Nov 2026"
INST=$(python -c "import json;print(','.join(x['ep'] for x in json.load(open('spec_set3.json'))['endpoints'] if x['family']=='QA3' and 'pool_' not in x['ep']))")
python lik.py --raw out/fake/set3/raw --out out/lik3 --records all --model Qwen/Qwen2.5-3B-Instruct --eps $INST --variants lora:out/lora/q3_alp3:0.3 --tag _inst
