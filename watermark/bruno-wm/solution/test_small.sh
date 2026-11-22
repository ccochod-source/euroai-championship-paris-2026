#!/bin/bash
W=/p/scratch/training2666/dasilvalopes3; cd $W/wm/solution; source $W/wm/kit/.venv/bin/activate
export WM_SYN=$W/syn_test HF_HOME=$W/.cache/hf TORCH_HOME=$W/.cache/torch XDG_CACHE_HOME=$W/.cache CUDA_VISIBLE_DEVICES=0
set -e
python gen_clean.py 300
python gen_masks.py
for m in trustmark stegastamp pixelseal; do python gen_wm.py $m a 2>&1 | grep -E "WM_DONE|failed|Error|Traceback" ; done
python - <<'PY'
import numpy as np, time, torch
from data import Synth
from PIL import Image
import os
for meth in ["trustmark","stegastamp","pixelseal"]:
    ds = Synth(meth); t=time.time(); fr=[]
    tiles=[]
    for i in range(48):
        x,y = ds[i]; fr.append(y.mean().item())
        if i<12:
            a=x.permute(1,2,0).numpy().copy(); m=np.kron(y.numpy(),np.ones((4,4)))>0.5
            a[m]=a[m]*0.5+np.array([0,255,0])*0.5; tiles.append(a.astype(np.uint8)[::2,::2])
    print(meth,"sample time %.3fs"%((time.time()-t)/48),"wm frac q",np.quantile(fr,[.1,.25,.5,.75,.9]).round(3))
    Image.fromarray(np.concatenate([np.concatenate(tiles[:6],1),np.concatenate(tiles[6:],1)],0)).save(f"{os.environ['WM_SYN']}/synth_{meth}.jpg",quality=85)
    c=np.load(f"{os.environ['WM_SYN']}/clean.npy",mmap_mode="r"); f=np.load(f"{os.environ['WM_SYN']}/{meth}_full_a.npy",mmap_mode="r")
    d=f[:50].astype(np.float32)-c[:50].astype(np.float32); print("  full PSNR %.1f dB"%(10*np.log10(255**2/(d**2).mean())))
PY
python train.py --method stegastamp --steps 150 --eval_every 75 --workers 16 --name test 2>&1 | grep -v -i warn | tail -8
echo TEST_DONE
