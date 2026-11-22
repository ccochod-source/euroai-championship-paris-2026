#!/bin/bash
# Installe la file GPU partagée et y bascule les tâches Model Equality en attente (exécuté sur le nœud de login).
set -e
Q=/p/scratch/training2666/cochod2/euroai-paris/gpu_queue
SIM=/p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
mkdir -p $Q
for g in 0 1 2 3; do mkdir -p $Q/gpu$g/running $Q/gpu$g/done $Q/gpu$g/failed $Q/gpu$g/logs; done
cp $SIM/gpu_worker.py $Q/gpu_worker.py
chmod -R g+rwX $Q; find $Q -type d -exec chmod g+s {} \;
HDR='#!/bin/bash
# TIMEOUT=7200
cd /p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
source venv/bin/activate
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
i=0
for t in $(ls $SIM/tasks/pending/*.sh 2>/dev/null | sort); do
  n=$(basename $t); g=$((i % 2)); i=$((i+1))
  { echo "$HDR"; cat $t; } > $Q/gpu$g/me_$n
  rm -f $t
done
END=$(date -d "$(squeue -j 15902386 -h -o %e)" +%s)
cat > $SIM/tasks/pending/00_gpu_worker.sh <<EOW
python3 $Q/gpu_worker.py --gpus 0,1,2,3 --until $END >> $Q/worker.log 2>&1
EOW
ls -R $Q | head -40
