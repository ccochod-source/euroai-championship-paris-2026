#!/bin/bash
# TIMEOUT=5400
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export SIM_SAMP_SMALL=1 HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_DATE="21 Nov 2026"
# 1) vérification des system prompts extraits : vraisemblance d'ep-29 et d'ep-08 AVEC leur system prompt
# 2) contrôle : poids de base ou variantes, toutes décodées en T = 0.9, top-k 50
python gen.py --protocol protocol70.jsonl --spec spec_set4.json --out out/fake/set4
for m in unsloth/Llama-3.2-1B-Instruct:LL1:ll1 Qwen/Qwen2.5-3B-Instruct:QA3:q3 Qwen/Qwen2.5-1.5B-Instruct:QB15:q15; do
  M=${m%%:*}; R=${m#*:}; F=${R%%:*}; T=${R##*:}
  E=$(python -c "import json;print(','.join(x['ep'] for x in json.load(open('spec_set4.json'))['endpoints'] if x['family']=='$F'))")
  python lik.py --raw out/fake/set4/raw --out out/lik4 --records all --model $M --eps $E --variants bf16 --samplers --tag _samp
  python lik.py --raw out/fake/set4/raw --out out/lik4 --records all --model $M --eps $E --variants bf16,fp8w,lora:out/lora/${T}_alp2:0.3
done
