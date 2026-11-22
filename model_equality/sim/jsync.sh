#!/usr/bin/env bash
# Synchronise avec JURECA (à lancer depuis ~/hackathon sur le VPS).
#   bash sim/jsync.sh push   : envoie le code sim/*.py|*.sh|*.jsonl|*.json et data/raw vers JURECA
#   bash sim/jsync.sh pull   : rapatrie sim/out/{lik_r2,lik_fake,design,distill,fake} depuis JURECA
S=/p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
case "$1" in
  push) rsync -a --include='*.py' --include='*.sh' --include='*.jsonl' --include='*.json' --exclude='*' sim/ jureca:$S/
        rsync -a data/raw/ jureca:$S/data/raw/ ;;
  pull) for d in lik_r2 lik_fake design distill fake lik; do mkdir -p sim/out/$d; rsync -a jureca:$S/out/$d/ sim/out/$d/; done ;;
  *) echo "usage: jsync.sh push|pull" ;;
esac
