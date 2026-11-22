#!/bin/bash
#SBATCH -J sim_lik2
#SBATCH -A training2666
#SBATCH -p dc-gpu
#SBATCH --reservation=cispahack
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%j.out
SIM=/p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
cd $SIM; source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
L="python lik.py --raw data/raw --out out/lik --records eps --model Qwen/Qwen2.5-3B-Instruct"
CUDA_VISIBLE_DEVICES=0 $L --variants fp8w,fp8wa > logs/lik2_g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $L --variants fp8ch,int8w,fp16 > logs/lik2_g1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $L --variants int4g128,bnb8 > logs/lik2_g2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $L --variants nf4 > logs/lik2_g3.log 2>&1 &
wait; echo ALL_DONE
