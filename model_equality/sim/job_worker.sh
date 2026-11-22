#!/bin/bash
#SBATCH -J sim_worker
#SBATCH -A training2666
#SBATCH -p dc-gpu
#SBATCH --reservation=cispahack
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
# Worker persistant : exécute les tâches tasks/pending/*.sh (ordre alphabétique), une par GPU libre.
SIM=/p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
cd $SIM; source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p tasks/pending tasks/running tasks/done
declare -A PID
end=$((SECONDS + 6*3600 - 600))
echo "worker start $(date +%T) on $(hostname)"
while [ $SECONDS -lt $end ] && [ ! -f tasks/STOP ]; do
  for g in 0 1 2 3; do
    p=${PID[$g]}
    if [ -n "$p" ] && kill -0 $p 2>/dev/null; then continue; fi
    t=$(ls tasks/pending/*.sh 2>/dev/null | sort | head -1)
    [ -z "$t" ] && continue
    name=$(basename $t .sh)
    mv $t tasks/running/$name.sh 2>/dev/null || continue
    echo "$(date +%T) gpu$g start $name"
    ( export CUDA_VISIBLE_DEVICES=$g; bash tasks/running/$name.sh > tasks/running/$name.log 2>&1; echo "exit=$?" >> tasks/running/$name.log
      mv tasks/running/$name.sh tasks/running/$name.log tasks/done/ ) &
    PID[$g]=$!
  done
  sleep 10
done
wait
echo "worker end $(date +%T)"
