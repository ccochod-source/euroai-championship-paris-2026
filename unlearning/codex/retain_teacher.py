"""Teacher trained ONLY on proxies, using early CLS+patch features.

12k train / 2k calibration / 2k untouched control. Forget images never train
the probe, select its regularization, or calibrate its temperature.
The original backbone has seen forget images; this is NOT a true gold teacher.
"""
import argparse
import json
from common import *


@torch.no_grad()
def features(model, images, blocks, views=('clean',), batch=100):
    out = {b:[] for b in blocks}
    for view in views:
        chunks = {b:[] for b in blocks}
        for start in range(0,len(images),batch):
            x = kit.preprocess(images[start:start+batch])
            if view == 'flip': x=x.flip(-1)
            with torch.autocast('cuda',dtype=torch.float16):
                h = model.norm_pre(model._pos_embed(model.patch_embed(x)))
                h = model.patch_drop(h)
                for b,block in enumerate(model.blocks):
                    h=block(h)
                    if b in blocks:
                        f=torch.cat([h[:,0],h[:,1:].mean(1)],-1)
                        chunks[b].append(f.float().cpu())
                    if b == max(blocks): break
        for b in blocks: out[b].append(torch.cat(chunks[b]))
    return {b:torch.stack(v).mean(0) for b,v in out.items()}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--blocks',default='6,7,8'); a=ap.parse_args()
    torch.set_num_threads(4); torch.manual_seed(41)
    assert torch.cuda.is_available(), 'Must run via gpu_queue'
    blocks=list(map(int,a.blocks.split(',')))
    (OUT/'cache').mkdir(exist_ok=True); (OUT/'logs').mkdir(exist_ok=True)
    proxy=torch.from_numpy(np.load(ROOT/'orehn/data/aigen16k.npz')['images'])
    zp=load(ROOT/'orehn/cache/proxy_logits.pt'); qp=(zp/2).softmax(-1)
    model=kit.load_model()
    fp=features(model,proxy[:14000],blocks)
    ff=features(model,kit.load_forget_images(),blocks,('clean','flip'))
    stats=[]; preds=[]; heads=[]
    for b in blocks:
        X=fp[b].cuda(); mu=X[:12000].mean(0); sd=X[:12000].std(0).clamp_min(.05)
        X=(X-mu)/sd; XF=(ff[b].cuda()-mu)/sd
        target=qp.cuda(); best=None
        for wd in [1e-4,1e-3,1e-2]:
            W=torch.zeros(X.shape[-1],100,device='cuda',requires_grad=True)
            bias=torch.zeros(100,device='cuda',requires_grad=True)
            opt=torch.optim.LBFGS([W,bias],lr=1,max_iter=100,history_size=12,line_search_fn='strong_wolfe')
            def closure():
                opt.zero_grad()
                z=X[:12000]@W+bias
                loss=-(target[:12000]*z.log_softmax(-1)).sum(-1).mean()+wd*W.square().sum()
                loss.backward(); return loss
            opt.step(closure)
            with torch.no_grad():
                z=X[12000:14000]@W+bias
                # Calibrate T2 logits against teacher T2 probabilities on proxies only.
                for temp in [.5,.75,1.,1.5,2.,3.]:
                    val=-(target[12000:14000]*(z/temp).log_softmax(-1)).sum(-1).mean().item()
                    if best is None or val<best['loss']:
                        best={'loss':val,'wd':wd,'temp':temp,'W':W.detach().clone(),'bias':bias.detach().clone()}
        with torch.no_grad():
            q=((XF@best['W']+best['bias'])/best['temp']).softmax(-1).cpu()
            zval=(X[12000:14000]@best['W']+best['bias'])/best['temp']
            metrics={'block':b,'wd':best['wd'],'temp':best['temp'],'cal_ce':best['loss'],
                     'cal_agree':(zval.argmax(-1).cpu()==zp[12000:14000].argmax(-1)).float().mean().item()}
            print(json.dumps(metrics),flush=True); stats.append(metrics); preds.append(q)
            heads.append({'block':b,'mu':mu.cpu(),'sd':sd.cpu(),
                          'W':best['W'].cpu(),'bias':best['bias'].cpu(),'temp':best['temp']})
    # Equal ensemble protects against selecting a depth with forget leakage.
    q=torch.stack(preds).mean(0)
    atomic_save(q,OUT/'cache/retain_teacher_q.pt')
    atomic_save(heads,OUT/'cache/retain_teacher_heads.pt')
    y=load(ROOT/'orehn/cache/teacher.pt')['y']
    losses=F.cross_entropy(2*q.clamp_min(1e-12).log(),y,reduction='none')
    report={'calibration':stats,'forget_implied_acc':(q.argmax(-1)==y).float().mean().item(),
            'forget_implied_mean':losses.mean().item(),'forget_implied_std':losses.std().item(),
            'caveat':'public synthetic proxies; not gold; frozen backbone was trained on forget'}
    (OUT/'logs/retain_teacher.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
