#!/usr/bin/env bash
# À lancer dès que les réponses de la réserve (prompt_id v3_*) sont dans data/raw :
#   bash sim/jsync.sh push && bash sim/dep_reserve.sh        (≈ 15 min de GPU, scripts courts sur gpu0 et gpu1)
# Recalcule la vraisemblance de TOUTES les réponses (anciennes + réserve) : bf16, fp8, proxys LoRA à force 0.3, et
# statistiques d'échantillonneur (grille réduite) ; sortie dans out/lik_res.
ssh jureca bash -s <<'EOF'
Q=/p/scratch/training2666/cochod2/euroai-paris/gpu_queue
H='#!/bin/bash
# TIMEOUT=3000
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_DATE="21 Nov 2026" SIM_FORCE=1 SIM_SAMP_SMALL=1
L="python lik.py --raw data/raw --out out/lik_res --records eps"
V () { echo "bf16,fp8w,lora:out/lora/$1_alpaca:0.3,lora:out/lora/$1_alp2:0.3,lora:out/lora/$1_med:0.3"; }'
QA='--model Qwen/Qwen2.5-3B-Instruct --eps ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33'
LL='--model unsloth/Llama-3.2-1B-Instruct --eps ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31'
QB='--model Qwen/Qwen2.5-1.5B-Instruct --eps ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32'
printf '%s\n$L %s --variants $(V q3)\n$L %s --variants bf16 --samplers --tag _samp\n' "$H" "$QA" "$QA" > $Q/gpu1/.tmp_r
printf '%s\n$L %s --variants $(V ll1)\n$L %s --variants $(V q15)\n$L %s --variants bf16 --samplers --tag _samp\n$L %s --variants bf16 --samplers --tag _samp\n' "$H" "$LL" "$QB" "$LL" "$QB" > $Q/gpu0/.tmp_r
mv $Q/gpu1/.tmp_r $Q/gpu1/me_01_reserve.sh; mv $Q/gpu0/.tmp_r $Q/gpu0/me_01_reserve.sh; ls $Q/gpu0 $Q/gpu1 | grep "\.sh"
EOF
