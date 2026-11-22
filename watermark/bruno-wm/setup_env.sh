#!/bin/bash
# Environment setup for the watermark task (run on the JURECA login node: needs internet)
set -eo pipefail
W=/p/scratch/training2666/dasilvalopes3
export UV_CACHE_DIR=$W/.uv_cache HF_HOME=$W/.cache/hf TORCH_HOME=$W/.cache/torch XDG_CACHE_HOME=$W/.cache PIP_CACHE_DIR=$W/.cache/pip
module load Stages/2026 >/dev/null 2>&1 || true
module load uv >/dev/null 2>&1 || true
cd $W/wm
[ -d kit ] || (mkdir kit && cd kit && unzip -q ../kit_review/starter_kit.zip)
cd kit
uv --version
[ -d .venv ] || uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python download_models.py
git -C models/videoseal rev-parse HEAD
echo ENV_DONE
