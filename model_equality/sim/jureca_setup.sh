#!/bin/bash
# Installation de l'environnement sim/ sur le nœud de login JURECA (internet OK ici, pas sur les nœuds de calcul).
set -e
SIM=/p/scratch/training2666/cochod2/euroai-paris/model_equality/sim
export HF_HOME=/p/scratch/training2666/cochod2/euroai-paris/.cache
export UV_CACHE_DIR=$SIM/.uvcache UV_PYTHON_INSTALL_DIR=$SIM/.uvpython PIP_CACHE_DIR=$SIM/.pipcache
cd $SIM
echo "[$(date +%T)] uv"
python3 -m pip install --quiet --target $SIM/uvbin uv
export PATH=$SIM/uvbin/bin:$PATH
echo "[$(date +%T)] venv"
uv venv --python 3.12 $SIM/venv
source $SIM/venv/bin/activate
echo "[$(date +%T)] pip"
uv pip install torch transformers peft accelerate bitsandbytes huggingface_hub numpy scipy scikit-learn pandas safetensors sentencepiece
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'tf', transformers.__version__, 'peft', peft.__version__)"
echo "[$(date +%T)] SETUP_DONE"
