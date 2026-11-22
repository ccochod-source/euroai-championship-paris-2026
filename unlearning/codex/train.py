"""Two new objectives: retain-only early teacher; bounded negative preference.

All artifacts remain in codex/. No submissions, Slurm jobs, or tokens.
"""
import argparse
import copy
import math
from common import *


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['retain_teacher','retain_teacher_online','npo'],required=True)
    ap.add_argument('--name',required=True); ap.add_argument('--steps',type=int,default=700)
    ap.add_argument('--lr',type=float,default=5e-5); ap.add_argument('--bs',type=int,default=40)
    ap.add_argument('--gamma',type=float,default=4.); ap.add_argument('--beta',type=float,default=.5)
    ap.add_argument('--cap',type=float,default=6.); ap.add_argument('--last',type=int,default=4)
    ap.add_argument('--full-eval',action='store_true'); ap.add_argument('--save',type=int,default=1)
    ap.add_argument('--seed',type=int,default=41)
    ap.add_argument('--checkpoint',default='',help='Optional existing state_dict to evaluate or continue; original remains the retain anchor')
    ap.add_argument('--alt-kl',type=float,default=0.,help='Preserve conditional distribution over non-label classes on forget views')
    ap.add_argument('--recovery',type=float,default=0.,help='Restore examples whose CE overshoots max(cap, reference CE)')
    ap.add_argument('--ref-conf',type=float,default=0.,help='Only apply negative preference where reference p(label) exceeds threshold')
    ap.add_argument('--extra-bright',type=float,default=0.,help='Fraction of training views devoted to broad contrast/brightness changes')
    a=ap.parse_args(); print(vars(a),flush=True)
    torch.set_num_threads(4); torch.manual_seed(a.seed)
    assert torch.cuda.is_available(), 'Must run via gpu_queue'
    for folder in ['logs','ckpt','cache']: (OUT/folder).mkdir(exist_ok=True)
    state=load(ROOT/'model.pt'); original=kit.load_model(); model=copy.deepcopy(original).eval()
    if a.checkpoint: model.load_state_dict(load(ROOT/a.checkpoint),strict=True)
    for p in original.parameters(): p.requires_grad_(False)
    for n,p in model.named_parameters():
        p.requires_grad_(a.last==12 or n.startswith(('norm.','head.')) or any(n.startswith(f'blocks.{b}.') for b in range(12-a.last,12)))
    params=[p for p in model.parameters() if p.requires_grad]
    imgs=kit.load_forget_images(); y=load(ROOT/'orehn/cache/teacher.pt')['y']
    proxy=torch.from_numpy(np.load(ROOT/'orehn/data/aigen16k.npz')['images'])
    zp=load(ROOT/'orehn/cache/proxy_logits.pt')
    q=load(OUT/'cache/retain_teacher_q.pt').cuda() if a.mode=='retain_teacher' else None
    heads=load(OUT/'cache/retain_teacher_heads.pt') if a.mode=='retain_teacher_online' else None
    if heads:
        heads=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in head.items()} for head in heads]
    @torch.no_grad()
    def online_target(x):
        h=original.norm_pre(original.patch_drop(original._pos_embed(original.patch_embed(x))))
        predictions=[]
        for b,block in enumerate(original.blocks):
            h=block(h)
            for head in heads:
                if head['block']==b:
                    f=torch.cat([h[:,0],h[:,1:].mean(1)],-1).float()
                    # Keep probe arithmetic float32; backbone follows server autocast.
                    with torch.autocast('cuda',enabled=False):
                        z=((f-head['mu'])/head['sd'])@head['W']+head['bias']
                        predictions.append((z/head['temp']).softmax(-1))
            if b==max(t['block'] for t in heads): break
        return torch.stack(predictions).mean(0)
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=0)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1,(s+1)/30)*(.2+.8*.5*(1+math.cos(math.pi*min(1,s/max(1,a.steps))))))
    scaler=torch.amp.GradScaler(); gen=torch.Generator(device='cuda').manual_seed(a.seed)
    t0=time.time()
    for step in range(1,a.steps+1):
        bi=torch.randint(len(imgs),(a.bs,)); pi=torch.randint(12000,(a.bs,))
        xclean=kit.preprocess(imgs[bi])
        xf=helpers.train_aug(xclean,gen)
        if a.extra_bright:
            choose=torch.rand(a.bs,device='cuda',generator=gen)<a.extra_bright
            contrast=.65+.6*torch.rand(a.bs,1,1,1,device='cuda',generator=gen)
            brightness=-.15+.3*torch.rand(a.bs,1,1,1,device='cuda',generator=gen)
            xf=torch.where(choose[:,None,None,None],(xclean*contrast+brightness).clamp(-1,1),xf)
        xp=kit.preprocess(proxy[pi])
        flip=torch.rand(a.bs,device='cuda',generator=gen)<.5
        xp=torch.where(flip[:,None,None,None],xp.flip(-1),xp)
        with torch.autocast('cuda',dtype=torch.float16):
            with torch.no_grad():
                zo=original(xp).float()
                zf0=original(xf).float() if a.mode=='npo' else None
                qview=online_target(xf) if a.mode=='retain_teacher_online' else None
            z=model(torch.cat([xf,xp])).float()
        if a.mode.startswith('retain_teacher'):
            lf=kl(qview if a.mode=='retain_teacher_online' else q[bi],z[:a.bs]).mean()
        else:
            yi=y[bi].cuda()
            lp=z[:a.bs].log_softmax(-1).gather(1,yi[:,None]).squeeze(-1)
            lp0=zf0.log_softmax(-1).gather(1,yi[:,None]).squeeze(-1)
            # Negative preference saturates as current likelihood falls below original.
            # The per-image cap explicitly prevents unbounded ascent.
            raw=2/a.beta*F.softplus(a.beta*(lp-lp0.detach()))
            ceiling=torch.maximum(torch.full_like(lp,a.cap),-lp0.detach()) if a.recovery else torch.full_like(lp,a.cap)
            active=(-lp<ceiling).float()*(lp0.detach().exp()>=a.ref_conf).float()
            lf=(active*raw).mean()
            if a.recovery:
                lf=lf+a.recovery*F.relu(-lp-ceiling).square().mean()
            if a.alt_kl:
                mask=F.one_hot(yi,num_classes=100).bool()
                alt_ref=(zf0.detach()/2).masked_fill(mask,-1e4).softmax(-1)
                alt_logp=(z[:a.bs]/2).masked_fill(mask,-1e4).log_softmax(-1)
                lf=lf+a.alt_kl*(alt_ref*(alt_ref.clamp_min(1e-12).log()-alt_logp)).sum(-1).mean()
        lr=kl((zo/2).softmax(-1),z[a.bs:]).mean()
        loss=lf+a.gamma*lr
        if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params,1.); scaler.step(opt); scaler.update(); sched.step()
        if step%100==0 or step==1:
            print(json.dumps({'step':step,'lf':lf.item(),'retain_KL_train':lr.item(),'secs':time.time()-t0}),flush=True)
        if step%350==0 and step!=a.steps:
            result=evaluate(model,state,imgs,proxy,zp,y,f'{a.name}_s{step}')
            append_result(f'{a.name}_s{step}',a.mode,result,note='diagnostic intermediate; no checkpoint; public synthetic control, not validation')
    result=evaluate(model,state,imgs,proxy,zp,y,a.name,a.full_eval)
    checkpoint=save_checkpoint(model,a.name) if a.save else ''
    append_result(a.name,a.mode,result,checkpoint,'public synthetic heldout control; true 70% gate unknown; no official score')


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        import csv
        import fcntl
        row=[time.strftime('%d/%m %H:%M'),'Codex','failed '+ ' '.join(sys.argv[1:]),'codex experiment']+['']*11+[repr(exc)]
        with (ROOT/'COORD/RESULTS.csv').open('a',newline='') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX); csv.writer(handle).writerow(row)
        raise
