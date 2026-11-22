"""Last-three-block distillation of calibrated OOF targets, with retain KL.

Fixed training budget, no leaderboard access. Own checkpoint and CPU JSON gate.
The evaluator is a different teacher (original early block8 retain-only probe).
"""
import argparse
import copy
import math
from common import *
from oof_views import TRAIN_VIEWS, EVAL_VIEWS, view


def centered(z): return z-z.mean(-1,keepdim=True)


@torch.no_grad()
def evaluate_views(model,data,images,proxy,zp,name):
    model.eval();y=data['labels'];zs={};report={'name':name,'views':{}}
    for v in EVAL_VIEWS:
        chunks=[]
        for begin in range(0,len(images),100):
            x=images[begin:begin+100].cuda().permute(0,3,1,2).float()/255
            with torch.autocast('cuda',dtype=torch.float16):chunks.append(model(view(x,v)*2-1).float().cpu())
        z=torch.cat(chunks);zs[v]=z
        losses=F.cross_entropy(z,y,reduction='none')
        q=data['evaluation_q'][v];gold_losses=F.cross_entropy(2*q.clamp_min(1e-12).log(),y,reduction='none')
        original=data['original'][v]
        orig_losses=F.cross_entropy(original,y,reduction='none')
        denom=(orig_losses.mean()-gold_losses.mean()).abs()+(orig_losses.std()-gold_losses.std()).abs()
        dist=(losses.mean()-gold_losses.mean()).abs()+(losses.std()-gold_losses.std()).abs()
        m=(1-dist/denom.clamp_min(1e-8)).clamp(0,1).item()
        k=kl(q,z).mean();k0=kl(q,original).mean()
        p=(1-k/k0.clamp_min(1e-8)).clamp(0,1).item()
        report['views'][v]={'acc':(z.argmax(-1)==y).float().mean().item(),'Lmean':losses.mean().item(),'Lstd':losses.std().item(),
                            'KL_eval_teacher':k.item(),'mean_std_proxy':m,'prediction_proxy':p,
                            'eval_teacher_mean_std':[gold_losses.mean().item(),gold_losses.std().item()]}
    ctl=torch.arange(14000,16000);zc=predict(model,proxy,ctl);z0=zp[ctl]
    agree=(zc.argmax(-1)==z0.argmax(-1)).float().mean().item();k=kl((z0/2).softmax(-1),zc).mean().item()
    report['control']={'agree':agree,'KL':k,'retain_est':agree/(1+k),
                       'source':'aigen16k[14000:16000], untouched by training; agreement proxy, no true labels'}
    alt=[v for v in EVAL_VIEWS if v!='clean']
    m=(report['views']['clean']['mean_std_proxy']+sum(report['views'][v]['mean_std_proxy'] for v in alt)/len(alt))/2
    p=(report['views']['clean']['prediction_proxy']+sum(report['views'][v]['prediction_proxy'] for v in alt)/len(alt))/2
    report['independent_proxy']=m*p*report['control']['retain_est']
    zs['control']=zc;zs['control_original']=z0;zs['labels']=y
    atomic_save(zs,OUT/'logs'/f'{name}_oof_logits.pt')
    (OUT/'logs'/f'{name}_oof_eval.json').write_text(json.dumps(report,indent=2))
    print('EVAL',json.dumps(report),flush=True)
    return report


def bootstrap_compare(data,baseline,candidate,rounds=300):
    y=data['labels'];n=len(y);gen=torch.Generator().manual_seed(9271)
    def metric(logits,idx):
        ms=[];ps=[]
        for v in EVAL_VIEWS:
            q=data['evaluation_q'][v][idx];z=logits[v][idx];o=data['original'][v][idx]
            l=F.cross_entropy(z,y[idx],reduction='none');g=F.cross_entropy(2*q.clamp_min(1e-12).log(),y[idx],reduction='none');lo=F.cross_entropy(o,y[idx],reduction='none')
            den=(lo.mean()-g.mean()).abs()+(lo.std()-g.std()).abs()
            d=(l.mean()-g.mean()).abs()+(l.std()-g.std()).abs()
            ms.append((1-d/den.clamp_min(1e-8)).clamp(0,1))
            ps.append((1-kl(q,z).mean()/kl(q,o).mean().clamp_min(1e-8)).clamp(0,1))
        return (ms[0]+torch.stack(ms[1:]).mean())/2*(ps[0]+torch.stack(ps[1:]).mean())/2
    diffs=[];retain=[]
    for _ in range(rounds):
        idx=torch.randperm(n,generator=gen)[:round(n*.3)]
        ci=torch.randint(2000,(2000,),generator=gen)
        r=[]
        for z in (baseline,candidate):
            a=(z['control'][ci].argmax(-1)==z['control_original'][ci].argmax(-1)).float().mean()
            k=kl((z['control_original'][ci]/2).softmax(-1),z['control'][ci]).mean()
            r.append(a/(1+k))
        diffs.append(metric(candidate,idx)*r[1]-metric(baseline,idx)*r[0]);retain.append(r[1])
    d=torch.stack(diffs);r=torch.stack(retain)
    return {'public_fraction_random_subsets':.3,'rounds':rounds,'proxy_delta_mean':d.mean().item(),
            'proxy_delta_95_interval':torch.quantile(d,torch.tensor([.025,.975])).tolist(),
            'retain_est_95_interval':torch.quantile(r,torch.tensor([.025,.975])).tolist()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='oof_calibrated_s1')
    ap.add_argument('--strength',type=float,default=1.);ap.add_argument('--steps',type=int,default=1100)
    ap.add_argument('--lr',type=float,default=3e-5);ap.add_argument('--gamma',type=float,default=2.)
    ap.add_argument('--baseline',default='orehn/ckpt/clem_A03r3_g4.pt')
    ap.add_argument('--targets',default='oof_targets.pt',help='OOF primary, or oof_adaptive_targets.pt priority2')
    a=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(4417)
    assert torch.cuda.is_available(),'Use only the existing GPU worker'
    data=load(OUT/'cache/oof_features.pt');target=load(OUT/'cache'/a.targets)
    images=kit.load_forget_images();proxy=torch.from_numpy(np.load(ROOT/'orehn/data/aigen16k.npz')['images']);zp=load(ROOT/'orehn/cache/proxy_logits.pt')
    original=kit.load_model().eval().requires_grad_(False)
    model=copy.deepcopy(original).eval()
    baseline_name='oof_ref_'+Path(a.baseline).stem
    baseline_log=OUT/'logs'/f'{baseline_name}_oof_logits.pt'
    if not baseline_log.exists():
        model.load_state_dict(load(ROOT/a.baseline),strict=True)
        evaluate_views(model,data,images,proxy,zp,baseline_name)
        model.load_state_dict(original.state_dict(),strict=True)
    for n,p in model.named_parameters():p.requires_grad_(n.startswith(('head.','norm.','blocks.9.','blocks.10.','blocks.11.')))
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=0)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1,(s+1)/30)*(.2+.8*.5*(1+math.cos(math.pi*min(s/a.steps,1)))))
    scaler=torch.amp.GradScaler();t0=time.time();bs=32
    for step in range(1,a.steps+1):
        bi=torch.randint(len(images),(bs,));pi=torch.randint(12000,(bs,));v=TRAIN_VIEWS[(step-1)%len(TRAIN_VIEWS)]
        xf=view(images[bi].cuda().permute(0,3,1,2).float()/255,v)*2-1
        xp=kit.preprocess(proxy[pi]);flip=torch.rand(bs,device='cuda')<.5
        xp=torch.where(flip[:,None,None,None],xp.flip(-1),xp)
        zt=centered(data['original'][v][bi])+a.strength*(target['targets'][v][bi]-centered(data['original'][v][bi]))
        zt=zt.cuda()
        with torch.autocast('cuda',dtype=torch.float16):
            with torch.no_grad():zo=original(xp).float()
            z=model(torch.cat([xf,xp])).float()
        # Centered logits avoid arbitrary offsets; both scoring temperatures retained.
        lf=.1*F.mse_loss(centered(z[:bs]),centered(zt))+.25*F.kl_div(z[:bs].log_softmax(-1),zt.softmax(-1),reduction='batchmean')+kl((zt/2).softmax(-1),z[:bs]).mean()
        lr=kl((zo/2).softmax(-1),z[bs:]).mean()+.01*F.mse_loss(centered(z[bs:]),centered(zo))
        loss=lf+a.gamma*lr
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params,1.);scaler.step(opt);scaler.update();sched.step()
        if step%100==0 or step==1:print(json.dumps({'step':step,'forget':lf.item(),'retain':lr.item(),'seconds':time.time()-t0}),flush=True)
    result=evaluate_views(model,data,images,proxy,zp,a.name)
    base=json.loads((OUT/'logs'/f'{baseline_name}_oof_eval.json').read_text())
    # Bootstrap is CPU2; GPU script remains alive only a short bounded period.
    candidate=load(OUT/'logs'/f'{a.name}_oof_logits.pt');baseline=load(baseline_log)
    boot=bootstrap_compare(data,baseline,candidate)
    held=['shift','rotate','noise']
    held_improved=all(result['views'][v]['KL_eval_teacher']<=base['views'][v]['KL_eval_teacher'] for v in held)
    eligible=result['control']['retain_est']>=.75 and boot['proxy_delta_95_interval'][0]>0 and held_improved
    ckpt=save_checkpoint(model,a.name) if eligible else ''
    patch=''
    if not eligible:
        # Shared scratch quota: keep only trained parameters for rejected trials.
        # A delta is explicitly NOT a valid strict=True submission checkpoint.
        delta={n:p.detach().cpu().float().contiguous() for n,p in model.named_parameters() if p.requires_grad}
        assert all(torch.isfinite(p).all() for p in delta.values())
        path=OUT/'ckpt'/f'{a.name}_delta.pt';atomic_save(delta,path);patch=str(path.relative_to(ROOT))
    gate={'name':a.name,'checkpoint':ckpt,'baseline':a.baseline,'config':vars(a),**boot,
          'retain_est':result['control']['retain_est'],'accuracy_estimate_type':'team proxy agreement/(1+KL); no actual validation labels',
          'heldout_view_KL_improved':held_improved,
          'local_candidate_eligible':eligible,'non_submission_parameter_delta':patch,
          'submitted':False,'submission_requirements':'fresh best-baseline comparison, current lock, 10min team cooldown, max3; never automatic in this script'}
    (OUT/'logs'/f'{a.name}_gate.json').write_text(json.dumps(gate,indent=2))
    summary={**result,'clean':result['views']['clean'],'bright':result['views']['bright']}
    append_result(a.name,'DINOv2 OOF 5x3; moment calibration; last3 + retainKL',summary,ckpt,json.dumps(gate))
    print('GATE',json.dumps(gate),flush=True)


if __name__=='__main__': main()
