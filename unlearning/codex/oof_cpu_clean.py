"""Bounded CPU2 fallback to start actual OOF fitting without waiting for GPU3.

Only the clean DINO feature pass runs on CPU. The worker extracts augmented
features separately. No CUDA visibility, scheduler calls, or leaderboard API.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import sys
import time
import torch
import numpy as np
import timm
from pathlib import Path
from timm.models.vision_transformer import checkpoint_filter_fn
from oof_fit import load,main as fit_main

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    assert not torch.cuda.is_available()
    dest=HERE/'cache/oof_features_clean.pt'
    if not dest.exists():
        m=timm.create_model('vit_base_patch14_dinov2',pretrained=False,img_size=224,num_classes=0).cpu().eval().requires_grad_(False)
        m.load_state_dict(checkpoint_filter_fn(load(HERE/'weights/dinov2_vitb14_pretrain.pth'),m),strict=True)
        images=np.load(ROOT/'forget_images.npz')['images'];out=[];start=time.time()
        mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
        with torch.no_grad():
            for begin in range(0,len(images),16):
                if (HERE/'cache/oof_features.pt').exists():
                    print('GPU feature cache completed first; stop CPU fallback',flush=True);return
                if time.time()-start>1200:raise TimeoutError('CPU fallback capped at20min; worker remains responsible for extraction')
                x=torch.from_numpy(images[begin:begin+16]).permute(0,3,1,2).float()/255
                out.append(m((x-mean)/std).float())
                if begin%160==0:print('CPU2_DINO',begin+len(x),'/',len(images),'seconds',round(time.time()-start,1),flush=True)
        features=torch.cat(out).half()
        heads=load(HERE/'cache/retain_teacher_heads.pt');ff=load(ROOT/'orehn/cache/blockfeats_clean.pt').float()
        qs={}
        for h in heads:
            f=ff[:,h['block']];z=((f-h['mu'])/h['sd'])@h['W']+h['bias'];qs[h['block']]=(z/h['temp']).softmax(-1)
        z=load(ROOT/'orehn/cache/orig_logits_views.pt')['clean']
        data={'features':{'clean':features},'original':{'clean':z},'labels':z.argmax(-1),
              'calibration_q':{'clean':(qs[6]+qs[7])/2},'evaluation_q':{'clean':qs[8]},
              'metadata':{'source':'CPU2 frozen public DINOv2; original clean block-feature cache; no GPU'}}
        torch.save(data,dest.with_suffix('.tmp'));dest.with_suffix('.tmp').replace(dest)
        print('CPU_CLEAN_READY',flush=True)
    sys.argv=['oof_fit.py','--features','oof_features_clean.pt','--targets-name','oof_targets_clean.pt']
    fit_main()
    print('CPU_OOF_15_HEADS_READY',flush=True)


if __name__=='__main__':main()
