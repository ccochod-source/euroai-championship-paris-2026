import torch, json
import torch.nn.functional as F
from checkpoint_changes import load
torch.manual_seed(0)
X=torch.load('loop/dino-forget.pt').float(); X=(X-X.mean(0))/X.std(0).clamp_min(1e-6)
c=torch.load('loop/gpu-cache-a.pt',weights_only=True)
w=load('model.pt'); W,b=w['head.weight'].float(),w['head.bias'].float(); del w
z=c['clean'].float()@W.T+b; y=z.argmax(-1)
print('classes',y.bincount(minlength=100).tolist())
print('orig conf',z.softmax(-1).max(-1).values.mean().item(),'orig loss',F.cross_entropy(z,y).item())
for v in ('flip','crop'):
    zv=c[v].float()@W.T+b; print(v,'agree',(zv.argmax(-1)==y).float().mean().item(),'loss',F.cross_entropy(zv,y).item())
perm=torch.randperm(len(X)); folds=perm.chunk(5); oof=torch.zeros(len(X),100)
for wd in (1e-3,1e-2,1e-1):
    for k,te in enumerate(folds):
        tr=torch.cat([f for j,f in enumerate(folds) if j!=k])
        A=torch.zeros(100,X.shape[1],requires_grad=True); B=torch.zeros(100,requires_grad=True)
        opt=torch.optim.LBFGS([A,B],max_iter=200,line_search_fn='strong_wolfe')
        def clo():
            opt.zero_grad(); l=F.cross_entropy(X[tr]@A.T+B,y[tr])+wd*A.square().sum(); l.backward(); return l
        opt.step(clo)
        oof[te]=(X[te]@A.T+B).detach()
    print('wd',wd,'oof acc',(oof.argmax(-1)==y).float().mean().item(),'oof loss',F.cross_entropy(oof,y).item(),'conf',oof.softmax(-1).max(-1).values.mean().item(),flush=True)
    torch.save({'oof':oof.clone(),'labels':y},f'loop/oof/oof-wd{wd}.pt')

