#!/usr/bin/env bash
# À lancer dès que le nœud GPU revient : vraisemblance des VRAIES réponses (v1 + v2a + v2b) en priorité (nom me_10_*).
#   bash sim/jsync.sh push && bash sim/dep_real.sh
ssh jureca bash -s <<'EOF'
Q=/p/scratch/training2666/cochod2/euroai-paris/gpu_queue
H='#!/bin/bash
# TIMEOUT=3600
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_DATE="21 Nov 2026" SIM_FORCE=1
L="python lik.py --raw data/raw --out out/lik_v2b"'
LL='$L --records eps --model unsloth/Llama-3.2-1B-Instruct --eps ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31'
QB='$L --records eps --model Qwen/Qwen2.5-1.5B-Instruct --eps ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32'
QA='$L --records eps --model Qwen/Qwen2.5-3B-Instruct --eps ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33'
printf '%s\n%s --variants bf16,fp8w,int8w,lora:out/lora/ll1_alpaca:1.0,lora:out/lora/ll1_med:1.0\n%s --variants bf16,fp8w,int8w,lora:out/lora/q15_alpaca:1.0,lora:out/lora/q15_med:1.0\n%s --variants bf16 --samplers --tag _samp\n%s --variants bf16 --samplers --tag _samp\n' "$H" "$LL" "$QB" "$LL" "$QB" > $Q/gpu0/.tmp_r
printf '%s\n%s --variants bf16,fp8w,int8w,lora:out/lora/q3_alpaca:1.0,lora:out/lora/q3_med:1.0\n%s --variants bf16 --samplers --tag _samp\n$L --records refs --model HuggingFaceTB/SmolLM2-360M-Instruct --variants bf16,fp8w,int8w,lora:out/lora/s360_alpaca:0.5\n' "$H" "$QA" "$QA" > $Q/gpu1/.tmp_r
mv $Q/gpu0/.tmp_r $Q/gpu0/me_10_real_v2b.sh; mv $Q/gpu1/.tmp_r $Q/gpu1/me_10_real_v2b.sh
cat $Q/gpu1/me_10_real_v2b.sh; ls $Q/gpu0 $Q/gpu1
EOF
