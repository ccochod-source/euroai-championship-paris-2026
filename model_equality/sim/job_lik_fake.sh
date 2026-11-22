#!/bin/bash
#SBATCH -J sim_likfake
#SBATCH -A training2666
#SBATCH -p dc-gpu
#SBATCH --reservation=cispahack
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
# Vraisemblance sur le FAUX jeu (vérité connue) : mesure la puissance des tests fp8/int8/nf4/excès.
SIM=/p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
cd $SIM; source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
V=bf16,fp8w,fp8wa,int8w,nf4
fams=$(python -c "
import json
s=json.load(open('spec_set1.json'))
for f in ['LL1','QB15','QA3']:
    e=[x for x in s['endpoints'] if x['family']==f]
    print(e[0]['base']+'|'+','.join(sorted(x['ep'] for x in e)))")
i=0
for fb in $fams; do
  base=${fb%%|*}; eps=${fb#*|}
  CUDA_VISIBLE_DEVICES=$i python lik.py --raw out/fake/set1/raw --out out/lik_fake --records eps --model $base --eps $eps --variants $V > logs/likf_g$i.log 2>&1 &
  i=$((i+1))
done
CUDA_VISIBLE_DEVICES=3 python lik.py --raw out/fake/refsim/raw --out out/lik_fake --records refs --model HuggingFaceTB/SmolLM2-360M-Instruct --variants $V > logs/likf_g3.log 2>&1 &
wait; echo ALL_DONE
