#!/bin/bash
# TIMEOUT=7200
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
wait_for () { while [ ! -f "$1" ]; do sleep 20; done; }
wait_for out/fake/set1/LL1.done
python lik.py --raw out/fake/set1/raw --out out/lik_fake --records eps --model unsloth/Llama-3.2-1B-Instruct --eps ep-01,ep-02,ep-05,ep-06,ep-07,ep-12,ep-14,ep-15,ep-16,ep-27,ep-33 --variants bf16,fp8w,fp8wa,int8w,nf4
