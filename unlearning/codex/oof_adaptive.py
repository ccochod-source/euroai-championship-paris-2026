"""Priority2: per-image confidence, with mass kept on original alternatives2-5.

Hardness is rank-averaged from DINO leave-one-out neighbours, augmentation drop,
and OOF pseudo-label confidence. More difficult => LOWER A (fixes the sign in
the prose proposal). Two parameters match the same unseen calibration moments.
CPU2 only; no public score or evaluation-teacher access.
"""
import json
import torch
import torch.nn.functional as F
from oof_fit import OUT,load,moments,center
from oof_views import TRAIN_VIEWS


def rank(x):
    # Stable average ranks for ties, rather than arbitrary order inside a tie.
    order=torch.argsort(x,stable=True);xs=x[order]
    unique,inverse,counts=torch.unique_consecutive(xs,return_inverse=True,return_counts=True)
    end=counts.cumsum(0).float();mid=(end-counts.float()/2)/len(x)
    result=torch.empty_like(x,dtype=torch.float);result[order]=mid[inverse]
    return result


def logits_for(A,orig,y):
    masked=orig.clone();masked[torch.arange(len(y)),y]=-1e4
    values,ids=masked.topk(4,-1)
    # Epsilon mass keeps every class possible for forward KL at scoring T2.
    alt=torch.full_like(orig,1e-5)
    alt.scatter_add_(1,ids,values.div(2).softmax(-1))
    alt[torch.arange(len(y)),y]=0;alt/=alt.sum(-1,keepdim=True)
    prob=alt*(1-A[:,None]);prob[torch.arange(len(y)),y]=A
    return center(prob.clamp_min(1e-12).log())


def main():
    torch.set_num_threads(2)
    data=load(OUT/'cache/oof_features.pt');bundle=load(OUT/'cache/oof_targets.pt');y=data['labels'];n=len(y)
    x=F.normalize(data['features']['clean'].float(),dim=-1)
    neighbours=[]
    for start in range(0,n,256):
        sim=x[start:start+256]@x.T
        sim[torch.arange(len(sim)),torch.arange(start,start+len(sim))]=-2
        neighbours.append((y[sim.topk(20,dim=-1).indices]!=y[start:start+len(sim),None]).float().mean(-1))
    knn=torch.cat(neighbours)
    p0=data['original']['clean'].softmax(-1)[torch.arange(n),y]
    paug=torch.stack([data['original'][v].softmax(-1)[torch.arange(n),y] for v in ('bright','crop','blur','color','jpeg')]).mean(0)
    poof=bundle['oof']['clean'].softmax(-1)[torch.arange(n),y]
    difficulty=(rank(knn)+rank(p0-paug)+rank(1-poof))/3
    targets={};report={'hardness':'mean ranks of DINO LOO-k20 wrong-label fraction, original confidence drop, 1-OOF confidence',
                       'direction':'A decreases with hardness, not increases','views':{}}
    params={}
    for v in data['original']:
        goal=moments(2*data['calibration_q'][v].clamp_min(1e-12).log(),y)
        if v in TRAIN_VIEWS:
            raw=torch.tensor([0.,1.],requires_grad=True)
            opt=torch.optim.LBFGS([raw],lr=.5,max_iter=100,line_search_fn='strong_wolfe')
            def make_A():return torch.sigmoid(raw[0]-F.softplus(raw[1])*(difficulty-.5)).clamp(1e-7,1-1e-7)
            def closure():
                opt.zero_grad();z=logits_for(make_A(),data['original'][v],y)
                loss=((moments(z,y)-goal)/goal.clamp_min(.2)).square().sum();loss.backward();return loss
            opt.step(closure);params[v]=raw.detach().clone()
        else:raw=params['clean']
        with torch.no_grad():
            A=torch.sigmoid(raw[0]-F.softplus(raw[1])*(difficulty-.5)).clamp(1e-7,1-1e-7)
            targets[v]=logits_for(A,data['original'][v],y)
            report['views'][v]={'goal':goal.tolist(),'achieved':moments(targets[v],y).tolist(),
                                'A_quantiles':torch.quantile(A,torch.tensor([0.,.1,.5,.9,1.])).tolist(),
                                'heldout_view':v not in TRAIN_VIEWS}
    torch.save({'targets':targets,'labels':y,'difficulty':difficulty,'metadata':report},OUT/'cache/oof_adaptive_targets.pt')
    (OUT/'logs/oof_adaptive.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
