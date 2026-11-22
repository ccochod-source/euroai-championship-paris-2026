#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=120
#SBATCH --output=/p/scratch/training2666/dasilvalopes3/wm/logs/runner_%j.out
# Generic task runner: 4 workers (one per GPU) execute the shell scripts dropped in queue/todo, oldest name first.
W=/p/scratch/training2666/dasilvalopes3; Q=$W/wm/queue; mkdir -p $Q/todo $Q/running $Q/done
cd $W/wm/solution; source $W/wm/kit/.venv/bin/activate
export HF_HOME=$W/.cache/hf TORCH_HOME=$W/.cache/torch XDG_CACHE_HOME=$W/.cache PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
worker() {
  g=$1
  while [ ! -f $Q/stop_$SLURM_JOB_ID ] && [ ! -f $Q/stop_all ]; do
    f=$(ls $Q/todo 2>/dev/null | sort | head -1)
    if [ -n "$f" ] && mv $Q/todo/$f $Q/running/$f 2>/dev/null; then
      echo "$(date +%T) job $SLURM_JOB_ID gpu $g start $f"
      CUDA_VISIBLE_DEVICES=$g bash $Q/running/$f > $W/wm/logs/task_${f%.sh}.log 2>&1
      echo "$(date +%T) job $SLURM_JOB_ID gpu $g done $f (exit $?)"; mv $Q/running/$f $Q/done/$f
    else sleep 3; fi
  done
}
for g in 0 1 2 3; do worker $g & sleep 1; done; wait
