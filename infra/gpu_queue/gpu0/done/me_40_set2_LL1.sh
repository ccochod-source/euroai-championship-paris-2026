#!/bin/bash
# TIMEOUT=5400
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python gen.py --protocol protocol2.jsonl --reps 2 --spec spec_set2.json --out out/fake/set2 --only-base unsloth/Llama-3.2-1B-Instruct
touch out/fake/set2/LL1.done
[ -f out/lora/ll1_alp2/adapter_config.json ] || python train_lora.py --model unsloth/Llama-3.2-1B-Instruct --data alpaca --seed 1 --steps 300 --out out/lora/ll1_alp2
EPS=$(python -c "import json;print(','.join(x['ep'] for x in json.load(open('spec_set2.json'))['endpoints'] if x['family']=='LL1'))")
python lik.py --raw out/fake/set2/raw --out out/lik_fake2 --records eps --model unsloth/Llama-3.2-1B-Instruct --eps $EPS --variants bf16,fp8w,fp8wa,int8w,fp16,lora:out/lora/ll1_alp2:0.5,lora:out/lora/ll1_alp2:1.0,lora:out/lora/ll1_med:0.5,lora:out/lora/ll1_alpaca:0.5
