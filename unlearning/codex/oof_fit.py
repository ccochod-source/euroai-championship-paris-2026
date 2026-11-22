"""Two-thread CPU OOF ensemble and moment calibration; never submits.

15 logistic probes: 5 stratified folds x 3 independent partitions, smoothing.1.
All views of an image remain in its fold. Only clean training features are fit.
Calibration uses retain-only probes6/7 as an explicit imperfect unseen surrogate;
probe8 is sealed for evaluation. Raw OOF moment matching is reported as a control.
"""
import argparse
import json
import os
import time
os.environ['CUDA_VISIBLE_DEVICES']=''  # This stage is CPU-only, even on a GPU-equipped login.
import torch
import torch.nn.functional as F
from pathlib import Path
from oof_views import TRAIN_VIEWS

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent


def load(p): return torch.load(p,map_location='cpu',weights_only=True)


def center(z): return z-z.mean(-1,keepdim=True)


def moments(z,y):
    l=F.cross_entropy(z,y,reduction='none')
    return torch.stack([l.mean(),l.std()])


def calibrated(z,orig,y,beta,temp,demote,strength=1.):
    zo=orig.clone()
    zo[torch.arange(len(y)),y]-=demote
    zt=beta*center(z)/temp+(1-beta)*center(zo)
    return center(orig)+(zt-center(orig))*strength


def fit_calibration(z,orig,y,goal):
    # Choose a stable, bounded solution, avoiding a solution driven by beta~0.
    z,orig,goal=z.double(),orig.double(),goal.double()
    solutions=[]
    for initial in ((.5,1.,15.),(.8,.7,20.),(.4,1.5,25.)):
        init=torch.tensor([(initial[0]-.25)/.65,(initial[1]-.25)/3.75,initial[2]/45],dtype=torch.float64)
        raw=torch.logit(init).requires_grad_()
        def physical():
            s=raw.sigmoid();return (.25+.65*s[0],.25+3.75*s[1],45*s[2])
        opt=torch.optim.LBFGS([raw],lr=1,max_iter=100,history_size=12,line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-11)
        def closure():
            opt.zero_grad();m=moments(calibrated(z,orig,y,*physical()),y)
            loss=((m-goal)/goal.clamp_min(.2)).square().sum();loss.backward();return loss
        opt.step(closure)
        with torch.no_grad():
            params=tuple(float(x) for x in physical())
            err=float(((moments(calibrated(z,orig,y,*params),y)-goal)/goal.clamp_min(.2)).square().sum())
        solutions.append((err,params))
    error,a=min(solutions,key=lambda q:q[0])
    return {'beta':float(a[0]),'temp':float(a[1]),'demote':float(a[2]),'relative_squared_error':error,
            'goal_mean_std':goal.tolist(),'achieved_mean_std':moments(calibrated(z,orig,y,*a),y).tolist()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--max-iter',type=int,default=100)
    ap.add_argument('--features',default='oof_features.pt')
    ap.add_argument('--targets-name',default='oof_targets.pt')
    ap.add_argument('--reuse-heads',action='store_true')
    a=ap.parse_args(); torch.set_num_threads(2);torch.manual_seed(4109)
    import fcntl
    lease=(OUT/'oof_fit.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX)
    data=load(OUT/'cache'/a.features);y=data['labels'];n=len(y);names=list(data['features'])
    accum={v:torch.zeros(n,100) for v in names}; individuals={v:[] for v in names};heads=[]
    started=time.time()
    # Fixed regularizer, no public-score or evaluator-driven selection.
    wd=1e-4
    reuse=a.reuse_heads or (a.features=='oof_features.pt' and (OUT/'cache/oof_targets_clean.pt').exists() and (OUT/'cache/oof_heads.pt').exists())
    previous=load(OUT/'cache/oof_heads.pt') if reuse else []
    for seed in (4109,7907,13513):
        pred={v:torch.empty(n,100) for v in names}
        # Stratification implemented locally to avoid modifying the shared venv.
        gen=torch.Generator().manual_seed(seed);assignment=torch.empty(n,dtype=torch.long)
        for c in y.unique():
            ci=torch.where(y==c)[0];ci=ci[torch.randperm(len(ci),generator=gen)]
            offset=int(torch.randint(5,(1,),generator=gen))
            assignment[ci]=(torch.arange(len(ci))+offset)%5
        for fold in range(5):
            fit=torch.where(assignment!=fold)[0];test=torch.where(assignment==fold)[0]
            if previous:
                saved=next(h for h in previous if h['seed']==seed and h['fold']==fold)
                assert torch.equal(test,saved['test_indices']) and torch.equal(fit,saved['fit_indices'])
                mu,sd,W,bias=(saved[k] for k in ('mu','sd','W','bias'))
            else:
                X=data['features']['clean'][fit].float()
                mu=X.mean(0);sd=X.std(0).clamp_min(.1);X=(X-mu)/sd
                W=torch.zeros(X.shape[1],100,requires_grad=True);bias=torch.zeros(100,requires_grad=True)
                opt=torch.optim.LBFGS([W,bias],lr=1,max_iter=a.max_iter,history_size=12,line_search_fn='strong_wolfe')
                def closure():
                    opt.zero_grad();z=X@W+bias
                    loss=F.cross_entropy(z,y[fit],label_smoothing=.1)+wd*W.square().sum()
                    loss.backward();return loss
                opt.step(closure)
            with torch.no_grad():
                for v in names:
                    pred[v][test]=center(((data['features'][v][test].float()-mu)/sd)@W+bias)
            heads.append({'seed':seed,'fold':fold,'fit_indices':fit,'test_indices':test,'mu':mu,'sd':sd,'W':W.detach(),'bias':bias.detach()})
            print(json.dumps({'seed':seed,'fold':fold,'held_ce':moments(pred['clean'][test],y[test]).tolist(),'seconds':time.time()-started}),flush=True)
        for v in names:
            # Ensemble probabilities at scoring T2, then return equivalent logits.
            accum[v]+=(pred[v]/2).softmax(-1)/3
            individuals[v].append(moments(pred[v],y))
    oof={v:center(2*accum[v].clamp_min(1e-12).log()) for v in names}
    report={'labels':'original clean argmax','folds':5,'seeds':[4109,7907,13513],'label_smoothing':.1,'weight_decay':wd,
            'training_views':'clean only, every image predicted exclusively by heads that excluded it',
            'target_stats':'unseen surrogate: retain-only probes6/7; NOT true gold',
            'evaluation_teacher':'retain-only probe8, no calibration or fitting using it','views':{}}
    targets={};params={}
    for v in names:
        goal=moments(2*data['calibration_q'][v].clamp_min(1e-12).log(),y)
        # Shift/rotation/noise remain unseen during training and calibration.
        p=fit_calibration(oof[v],data['original'][v],y,goal) if v in TRAIN_VIEWS else dict(params['clean'])
        params[v]=p
        targets[v]=calibrated(oof[v],data['original'][v],y,p['beta'],p['temp'],p['demote'])
        report['views'][v]={'raw_oof_mean_std':moments(oof[v],y).tolist(),
                            'individual_unseen_mean_std':torch.stack(individuals[v]).mean(0).tolist(),
                            'oof_accuracy':(oof[v].argmax(-1)==y).float().mean().item(),
                            'heldout_view':v not in TRAIN_VIEWS,**p}
    # In-sample labels are NEVER used to predict their own row, assert head exclusion.
    for h in heads: assert not torch.isin(h['test_indices'],h['fit_indices']).any()
    out={'targets':targets,'oof':oof,'params':params,'labels':y,'metadata':report}
    dest=OUT/'cache'/a.targets_name
    torch.save(out,dest.with_suffix('.tmp'));dest.with_suffix('.tmp').replace(dest)
    torch.save(heads,OUT/'cache/oof_heads.pt')
    (OUT/'logs'/('oof_calibration_clean.json' if len(names)==1 else 'oof_calibration.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
