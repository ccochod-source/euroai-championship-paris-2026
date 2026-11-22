import torch, json
import torch.nn.functional as F
from checkpoint_changes import load
torch.manual_seed(1)
c=torch.load('loop/gpu-cache-a.pt',weights_only=True); c={k:v.float() for k,v in c.items()}
P=torch.load('loop/tiny-imagenet-control-2.pt',weights_only=True).float(); Pa,Pb=P[:200],P[200:]
w=load('model.pt'); W0,b0=w['head.weight'].float(),w['head.bias'].float(); del w
t=torch.load('loop/oof/oof-wd0.01.pt',weights_only=True); T,y=t['oof'].float(),t['labels']
perm=torch.randperm(3500); held,fit=perm[:700],perm[700:]
def kl2(tl,z): return F.kl_div((z/2).log_softmax(-1),(tl/2).softmax(-1),reduction='batchmean').item()
base=c['clean']@W0.T+b0; pb=Pb@W0.T+b0; pa=Pa@W0.T+b0
def rep(W,b):
    z=c['clean']@W.T+b; l=F.cross_entropy(z[fit],y[fit],reduction='none'); zp=Pb@W.T+b
    return dict(loss=round(l.mean().item(),3),std=round(l.std().item(),3),kl_t=round(kl2(T[fit],z[fit]),3),
      held_kl_t=round(kl2(T[held],z[held]),3),held_loss=round(F.cross_entropy(z[held],y[held]).item(),3),
      pub_agree=round((zp.argmax(-1)==pb.argmax(-1)).float().mean().item(),3),pub_kl=round(kl2(pb,zp),3))
for name in ('loop/head-1.1.pt','loop/temperature-3.789.pt'):
    h=load(name); print(name,rep(h['head.weight'].float(),h['head.bias'].float()))
X=torch.cat([c[v][fit] for v in ('clean','flip','crop')]); G=T[fit].repeat(3,1)
for lam in (0.0,3.0,10.0,30.0):
    W=W0.clone().requires_grad_(True); b=b0.clone().requires_grad_(True); o=torch.optim.Adam([W,b],lr=1e-3)
    for s in range(1,801):
        i=torch.randint(len(X),(256,)); z=X[i]@W.T+b
        loss=F.kl_div(z.log_softmax(-1),G[i].softmax(-1),reduction='batchmean')+4*F.kl_div((z/2).log_softmax(-1),(G[i]/2).softmax(-1),reduction='batchmean')
        if lam: loss=loss+lam*F.kl_div(((Pa@W.T+b)/2).log_softmax(-1),(pa/2).softmax(-1),reduction='batchmean')
        o.zero_grad(); loss.backward(); o.step()
        if s in (100,400,800):
            print(lam,s,rep(W.detach(),b.detach()),flush=True)
            torch.save({'head.weight':W.detach().clone(),'head.bias':b.detach().clone()},f'loop/oof/head-lam{lam}-s{s}.pt')

