"""GPU-worker-only extraction: DINOv2 features + independent retain-only probes.

Features come from the public frozen DINOv2-B/14. Calibration probes 6/7 and
evaluation probe 8 were fitted exclusively on non-forget proxy images earlier.
None of these probes is a true gold model. Writes only codex/.
"""
import time
import timm
from timm.models.vision_transformer import checkpoint_filter_fn
from common import *
from oof_views import EVAL_VIEWS, view


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'Use the existing GPU worker'
    p=OUT/'cache/oof_features.pt'
    if p.exists(): print('FEATURE_CACHE_EXISTS',p,flush=True);return
    dino=timm.create_model('vit_base_patch14_dinov2',pretrained=False,img_size=224,num_classes=0)
    dino.load_state_dict(checkpoint_filter_fn(load(OUT/'weights/dinov2_vitb14_pretrain.pth'),dino),strict=True)
    dino=dino.cuda().eval().requires_grad_(False)
    orig=kit.load_model().eval().requires_grad_(False)
    heads=load(OUT/'cache/retain_teacher_heads.pt')
    heads=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in h.items()} for h in heads]
    images=kit.load_forget_images()
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    result={'features':{},'original':{},'calibration_q':{},'evaluation_q':{},
            'metadata':{'backbone':'public DINOv2-B14, positional embeddings resized to224',
            'weights_url':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
            'calibration_teacher':'retain-only original probes blocks6/7, never fitted on forget labels',
            'evaluation_teacher':'separate retain-only original probe block8, excluded from target calibration',
            'gold_available':False}}
    started=time.time()
    for name in EVAL_VIEWS:
        fs,zs,cs,es=[],[],[],[]
        for begin in range(0,len(images),80):
            x=images[begin:begin+80].cuda().permute(0,3,1,2).float()/255
            x=view(x,name)
            with torch.autocast('cuda',dtype=torch.float16):
                fs.append(dino((x-mean)/std).float().cpu())
                h=orig.norm_pre(orig.patch_drop(orig._pos_embed(orig.patch_embed(x*2-1))))
                probs={}
                for b,block in enumerate(orig.blocks):
                    h=block(h)
                    for head in heads:
                        if head['block']==b:
                            f=torch.cat([h[:,0],h[:,1:].mean(1)],-1).float()
                            with torch.autocast('cuda',enabled=False):
                                z=((f-head['mu'])/head['sd'])@head['W']+head['bias']
                                probs[b]=(z/head['temp']).softmax(-1)
                zs.append(orig.forward_head(orig.norm(h)).float().cpu())
                cs.append(((probs[6]+probs[7])/2).cpu())
                es.append(probs[8].cpu())
        result['features'][name]=torch.cat(fs).half()
        result['original'][name]=torch.cat(zs)
        result['calibration_q'][name]=torch.cat(cs)
        result['evaluation_q'][name]=torch.cat(es)
        print('EXTRACT',name,'seconds',round(time.time()-started,1),flush=True)
    result['labels']=result['original']['clean'].argmax(-1)
    expected=load(ROOT/'orehn/cache/teacher.pt')['y']
    assert torch.equal(result['labels'],expected), 'Original forward or image order mismatch'
    atomic_save(result,p)
    print('FEATURE_CACHE_READY',p,p.stat().st_size,flush=True)


if __name__=='__main__': main()
