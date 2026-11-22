#!/bin/bash
#SBATCH -J sim_design
#SBATCH -A training2666
#SBATCH -p dc-gpu
#SBATCH --reservation=cispahack
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
SIM=/p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
cd $SIM; source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
D="python kl_design.py --prompts design_prompts.jsonl --n 3 --temp 0.7"
CUDA_VISIBLE_DEVICES=0 $D --model Qwen/Qwen2.5-1.5B-Instruct --variants fp8w,fp8wa,int8w,nf4,lora:out/lora/q15_med:0.3 --out out/design/q15.jsonl > logs/design_g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $D --model Qwen/Qwen2.5-3B-Instruct --variants fp8w,fp8wa,int8w,nf4 --out out/design/q3.jsonl > logs/design_g1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $D --model unsloth/Llama-3.2-1B-Instruct --variants fp8w,fp8wa,int8w,nf4,lora:out/lora/ll1_alpaca:0.3 --out out/design/ll1.jsonl > logs/design_g2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $D --model HuggingFaceTB/SmolLM2-360M-Instruct --variants fp8w,fp8wa,int8w,nf4,lora:out/lora/s360_alpaca:0.3 --out out/design/s360.jsonl > logs/design_g3.log 2>&1 &
wait; echo ALL_DONE
