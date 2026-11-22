#!/bin/bash
# TIMEOUT=5400
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_DATE="21 Nov 2026"
D () { [ -f out/distill3/$2_$3_s$4.json ] || python distill.py --model $1 --raw data/raw --train-ep $3 --eval-eps $5 --steps $4 --out out/distill3/$2_$3_s$4.json; }
D HuggingFaceTB/SmolLM2-360M-Instruct REF ref-07 25 ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09
D HuggingFaceTB/SmolLM2-360M-Instruct REF ref-04 25 ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09
D Qwen/Qwen2.5-3B-Instruct QA ep-24 25 ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33
D Qwen/Qwen2.5-3B-Instruct QA ep-01 25 ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33
D Qwen/Qwen2.5-3B-Instruct QA ep-18 25 ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33
D Qwen/Qwen2.5-3B-Instruct QA ep-17 25 ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33
D Qwen/Qwen2.5-3B-Instruct QA ep-12 25 ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33
D Qwen/Qwen2.5-3B-Instruct QA ep-25 25 ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33
