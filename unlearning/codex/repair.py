"""Retain repair anchored to an already-unlearned source on clean/altered forget.

New direction: recover utility AFTER forgetting, without inventing a new gold
teacher. Local acceptance is decided before looking at any new public score.
This script never submits, reads credentials, or invokes Slurm.
"""
import argparse
import copy
import math
from common import *

VIEWS=['clean','bright','flip','crop','shift','noise','blur']


def wilson_lower(success,total,z=1.96):
    if total==0:return 0.
    p=success/total
    return (p+z*z/(2*total)-z*math.sqrt(p*(1-p)/total+z*z/(4*total*total)))/(1+z*z/total)


def summarize_gate(name,reference_name,q):
    result=json.loads((OUT/'logs'/f'{name}.json').read_text())
    base=json.loads((OUT/'logs'/f'{reference_name}.json').read_text())
    Z=load(OUT/'logs'/f'{name}_logits.pt'); B=load(OUT/'logs'/f'{reference_name}_logits.pt')
    assert torch.equal(Z['indices'],B['indices'])
    idx=Z['indices']; comparisons={}
    for view in VIEWS:
        src=(B[view]/2).softmax(-1)
        comparisons[view]={
            'source_KL':kl(src,Z[view]).mean().item(),
            'mean_loss_delta':result[view]['Lmean']-base[view]['Lmean'],
            'std_loss_delta':result[view]['Lstd']-base[view]['Lstd'],
            'KL_fixed_target':kl(q[idx],Z[view]).mean().item(),
            'KL_fixed_target_source':kl(q[idx],B[view]).mean().item()}
    ctl=result['control']; bctl=base['control']
    retention_proxy=ctl['agree']/(1+ctl['KL'])
    # Labels are absent. This is explicitly NOT an accuracy certificate.
    agreement_lower=wilson_lower(round(ctl['agree']*2000),2000)
    conservative_proxy=agreement_lower/(1+ctl['KL'])
    preserved=all(v['source_KL']<=.02 and abs(v['mean_loss_delta'])<=.15 and abs(v['std_loss_delta'])<=.15
                  and v['KL_fixed_target']<=v['KL_fixed_target_source']+.02 for v in comparisons.values())
    improved=ctl['KL']<bctl['KL']-.01 and ctl['agree']>=bctl['agree']
    report={'name':name,'source':reference_name,'comparison':comparisons,
            'retention_proxy':retention_proxy,'retention_proxy_conservative':conservative_proxy,
            'forget_preserved_locally':preserved,'retain_improved_locally':improved,
            'proxy_above_75':conservative_proxy>=.75,
            'local_accuracy_certified':False,'autonomous_submission_eligible':False,
            'decision':'no autonomous submission: true labeled validation unavailable; proxy is not accuracy'}
    (OUT/'logs'/f'{name}_gate.json').write_text(json.dumps(report,indent=2))
    print('LOCAL_GATE',json.dumps(report),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--name',required=True)
    ap.add_argument('--steps',type=int,default=900); ap.add_argument('--lr',type=float,default=2e-5)
    ap.add_argument('--last',type=int,default=4); ap.add_argument('--anchor',type=float,default=8)
    ap.add_argument('--bs',type=int,default=40); ap.add_argument('--seed',type=int,default=73)
    a=ap.parse_args(); print(vars(a),flush=True)
    torch.set_num_threads(4); torch.manual_seed(a.seed)
    assert torch.cuda.is_available(),'GPU queue only'
    for folder in ['logs','ckpt','cache']:(OUT/folder).mkdir(exist_ok=True)
    assert Path(a.source).parts[0] in ['orehn','codex'] and '..' not in Path(a.source).parts
    state=load(ROOT/'model.pt'); original=kit.load_model()
    source=copy.deepcopy(original); source.load_state_dict(load(ROOT/a.source),strict=True); source.eval()
    model=copy.deepcopy(source).eval()
    for frozen in [original,source]:
        for p in frozen.parameters():p.requires_grad_(False)
    for n,p in model.named_parameters():
        p.requires_grad_(a.last==12 or n.startswith(('norm.','head.')) or any(n.startswith(f'blocks.{b}.') for b in range(12-a.last,12)))
    params=[p for p in model.parameters() if p.requires_grad]
    imgs=kit.load_forget_images(); T=load(ROOT/'orehn/cache/teacher.pt'); y=T['y']
    proxy=torch.from_numpy(np.load(ROOT/'orehn/data/aigen16k.npz')['images']); zp=load(ROOT/'orehn/cache/proxy_logits.pt')
    source_stem=Path(a.source).stem
    cfg_path=ROOT/'orehn/logs'/f'{source_stem}.json'
    cfg=json.loads(cfg_path.read_text())['args'] if cfg_path.exists() else {'A':.15,'alpha':4.,'rpow':1.}
    q=helpers.make_targets(T,cfg['A'],cfg['alpha'],rpow=cfg.get('rpow',1),verbose=False)
    refname=f'ref_{source_stem}_night'
    if not (OUT/'logs'/f'{refname}_logits.pt').exists():
        baseline=evaluate(source,state,imgs,proxy,zp,y,refname,True)
        append_result(refname,'night source reference',baseline,a.source,'no submission; full seven-view reference; unlabeled control')
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=0)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1,(s+1)/30)*(.1+.9*.5*(1+math.cos(math.pi*min(1,s/max(1,a.steps))))))
    scaler=torch.amp.GradScaler(); g=torch.Generator(device='cuda').manual_seed(a.seed)
    t0=time.time()
    for step in range(1,a.steps+1):
        bi=torch.randint(len(imgs),(a.bs,)); pi=torch.randint(12000,(a.bs,))
        xf=helpers.train_aug(kit.preprocess(imgs[bi]),g)
        # Explicitly represent mild altered groups in replay, in addition to random views.
        kind=step%7
        if kind==1:xf=kit.preprocess(imgs[bi])*.8+.1
        elif kind==2:xf=kit.preprocess(imgs[bi]).flip(-1)
        elif kind==3:
            raw=kit.preprocess(imgs[bi]);xf=F.interpolate(raw[...,13:-13,13:-13],size=224,mode='bilinear',align_corners=False)
        elif kind==4:xf=F.pad(kit.preprocess(imgs[bi]),(8,8,8,8),mode='reflect')[...,5:229,11:235]
        xp=helpers.train_aug(kit.preprocess(proxy[pi]),g,strength=.4)
        with torch.autocast('cuda',dtype=torch.float16):
            with torch.no_grad():zs=source(xf).float(); zo=original(xp).float()
            z=model(torch.cat([xf,xp])).float()
        lf=kl((zs/2).softmax(-1),z[:a.bs]).mean()
        lr=kl((zo/2).softmax(-1),z[a.bs:]).mean()
        loss=a.anchor*lf+lr
        if not torch.isfinite(loss):raise RuntimeError('nonfinite repair loss')
        opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params,1.);scaler.step(opt);scaler.update();sched.step()
        if step%100==0 or step==1:print(json.dumps({'step':step,'forget_source_KL':lf.item(),'retain_original_KL':lr.item(),'secs':time.time()-t0}),flush=True)
    result=evaluate(model,state,imgs,proxy,zp,y,a.name,True)
    gate=summarize_gate(a.name,refname,q)
    # Persist meaningful locally improved repairs only; failed experiments retain logits/reports.
    path=save_checkpoint(model,a.name) if gate['retain_improved_locally'] and gate['forget_preserved_locally'] else ''
    append_result(a.name,'retain repair after forgetting',result,path,
                  '7 views; source='+a.source+'; local_preserved='+str(gate['forget_preserved_locally'])+
                  '; retain_improved='+str(gate['retain_improved_locally'])+'; conservative_retention_proxy='+str(round(gate['retention_proxy_conservative'],4))+
                  '; accuracy NOT certified; no submission')
    (OUT/'logs'/f'{a.name}_done.json').write_text(json.dumps({'args':vars(a),'checkpoint':path,'gate':gate},indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import fcntl
        with (ROOT/'COORD/RESULTS.csv').open('a',newline='') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            csv.writer(handle).writerow([time.strftime('%d/%m %H:%M'),'Codex','repair failed','night repair']+['']*11+[repr(exc)])
        raise
