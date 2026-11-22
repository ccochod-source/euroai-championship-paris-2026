#!/bin/bash
# TIMEOUT=5400
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python gen.py --protocol protocol2.jsonl --reps 2 --spec spec_set2.json --out out/fake/set2 --only-base Qwen/Qwen2.5-3B-Instruct
touch out/fake/set2/QA3.done
[ -f out/lora/q3_alp2/adapter_config.json ] || python train_lora.py --model Qwen/Qwen2.5-3B-Instruct --data alpaca --seed 1 --steps 300 --out out/lora/q3_alp2
EPS=$(python -c "import json;print(','.join(x['ep'] for x in json.load(open('spec_set2.json'))['endpoints'] if x['family']=='QA3'))")
python lik.py --raw out/fake/set2/raw --out out/lik_fake2 --records eps --model Qwen/Qwen2.5-3B-Instruct --eps $EPS --variants bf16,fp8w,fp8wa,int8w,fp16,lora:out/lora/q3_alp2:0.5,lora:out/lora/q3_alp2:1.0,lora:out/lora/q3_med:0.5,lora:out/lora/q3_alpaca:0.5
