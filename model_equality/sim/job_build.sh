#!/bin/bash
#SBATCH -J sim_build
#SBATCH -A training2666
#SBATCH -p dc-gpu
#SBATCH --reservation=cispahack
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
SIM=/p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
cd $SIM; source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TR="python train_lora.py"; G="python gen.py --protocol protocol.jsonl --reps 2"
run_fam () {  # $1 gpu  $2 base  $3 tag
  export CUDA_VISIBLE_DEVICES=$1
  [ -f out/lora/$3_alpaca/adapter_config.json ] || $TR --model $2 --data alpaca --steps 300 --out out/lora/$3_alpaca
  [ -f out/lora/$3_med/adapter_config.json ] || $TR --model $2 --data med --steps 200 --out out/lora/$3_med
  $G --spec spec_set1.json --out out/fake/set1 --only-base $2
}
run_fam 0 Qwen/Qwen2.5-0.5B-Instruct q05 > logs/build_g0.log 2>&1 &
run_fam 1 Qwen/Qwen2.5-1.5B-Instruct q15 > logs/build_g1.log 2>&1 &
run_fam 2 unsloth/Llama-3.2-1B-Instruct ll1 > logs/build_g2.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=3
  [ -f out/lora/s360_alpaca/adapter_config.json ] || $TR --model HuggingFaceTB/SmolLM2-360M-Instruct --data alpaca --steps 300 --out out/lora/s360_alpaca
  $G --spec spec_refsim.json --out out/fake/refsim ) > logs/build_g3.log 2>&1 &
wait
echo ALL_DONE
