#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=/p/scratch/training2666/dasilvalopes3/wm/logs/one_%x_%j.out
# usage: sbatch -J <method> job_one.sh   (env: NAME, STEPS)
W=/p/scratch/training2666/dasilvalopes3; cd $W/wm/solution; source $W/wm/kit/.venv/bin/activate
export HF_HOME=$W/.cache/hf TORCH_HOME=$W/.cache/torch XDG_CACHE_HOME=$W/.cache PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=0
M=$SLURM_JOB_NAME; L=$W/wm/logs
[ -f $W/syn/${M}_msgs_a.npy ] || python gen_wm.py $M a > $L/gen_$M.log 2>&1
python train.py --method $M --steps ${STEPS:-20000} --workers 28 --name ${NAME:-v1} > $L/train_${M}_${NAME:-v1}.log 2>&1
