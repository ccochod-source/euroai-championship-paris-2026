#!/bin/bash
# usage: q.sh <gpu> <jobname> <timeout> <python args...>   (run on jureca login)
Q=/p/scratch/training2666/cochod2/euroai-paris/gpu_queue; G=$1; J=$2; TO=$3; shift 3
cat > $Q/gpu$G/.tmp_$J <<EOT
#!/bin/bash
# TIMEOUT=$TO
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning
source .venv/bin/activate
python -u orehn/src/$@
EOT
mv $Q/gpu$G/.tmp_$J $Q/gpu$G/$J.sh; echo queued $J on gpu$G
