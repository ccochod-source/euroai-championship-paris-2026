from common import *
model = kit.load_model(); imgs = kit.load_forget_images(); N = len(imgs)
z0 = torch.load(C + "/orig_logits_views.pt")["clean"]; y = z0.argmax(1)
g = torch.Generator(device="cuda").manual_seed(1)
R = torch.zeros(N, 100); Py = []; 
with torch.no_grad():
    for kind in PERTS:
        out = []
        for i in range(0, N, 125):
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(model(perturb(kit.preprocess(imgs[i:i+125]), kind, g)).float().cpu())
        z = torch.cat(out); p2 = (z / 2).softmax(1); R += p2 / len(PERTS); Py.append(p2[torch.arange(N), y])
        print(f"{kind:8s} agree={(z.argmax(1)==y).float().mean():.3f} T2 p_y={Py[-1].mean():.3f} T1 loss={F.cross_entropy(z,y):.3f}", flush=True)
Py = torch.stack(Py, 1)
# CV logistic probes on CLS of blocks 8,9,10
f = torch.load(C + "/blockfeats_clean.pt").float().cuda(); yc = y.cuda()
perm = torch.randperm(N, generator=torch.Generator().manual_seed(0)); fold = torch.empty(N, dtype=torch.long); fold[perm] = torch.arange(N) % 5; fold = fold.cuda()
def cvlog(X, wd):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6); P = torch.zeros(N, 100, device="cuda")
    for k in range(5):
        tr, te = fold != k, fold == k
        W = torch.zeros(X.shape[1], 100, device="cuda", requires_grad=True); b = torch.zeros(100, device="cuda", requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=200, history_size=20, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); l = F.cross_entropy(X[tr] @ W + b, yc[tr]) + wd * W.pow(2).sum(); l.backward(); return l
        opt.step(closure)
        with torch.no_grad(): P[te] = (X[te] @ W + b).softmax(1)
    return P.cpu()
probes = {}
for l in (7, 8, 9, 10):
    best = None
    for wd in (1e-4, 1e-3, 1e-2):
        P = cvlog(f[:, l, :768], wd); nll = -P[torch.arange(N), y].clamp_min(1e-9).log().mean().item()
        if best is None or nll < best[0]: best = (nll, wd, P)
    probes[l] = best[2]; print(f"block{l} CV logistic: acc={(best[2].argmax(1)==y).float().mean():.4f} nll={best[0]:.4f} wd={best[1]}", flush=True)
def lo(p): p = p.clamp(1e-4, 1 - 1e-4); return (p / (1 - p)).log()
zs = lambda v: (v - v.mean()) / v.std()
idx = torch.arange(N)
comps = {"pert": zs(lo(Py).mean(1)), "p8": zs(lo(probes[8][idx, y])), "p9": zs(lo(probes[9][idx, y])), "p10": zs(lo(probes[10][idx, y]))}
M = torch.stack(list(comps.values()), 1); print("corr between hardness components\n", torch.corrcoef(M.T))
s = zs(M.mean(1))
torch.save({"y": y, "R": R, "Py": Py, "P8": probes[8], "P9": probes[9], "P10": probes[10], "s": s, "comps": comps, "fold": fold.cpu()}, C + "/teacher.pt")
T = torch.load(C + "/teacher.pt")
for A in (0.9, 0.8, 0.7, 0.6):
    for alpha in (1.5, 2.5, 4.0): make_targets(T, A, alpha)
# proxies stats
import numpy as np, os
if os.path.exists(ROOT + "/orehn/data/aigen16k.npz"):
    P = torch.from_numpy(np.load(ROOT + "/orehn/data/aigen16k.npz")["images"]); zp = kit.predict(model, P, 125); torch.save(zp, C + "/proxy_logits.pt")
    c = zp.softmax(1).max(1).values; bc = torch.bincount(zp.argmax(1), minlength=100)
    print("proxy: n", len(P), "conf mean", c.mean().item(), "median", c.median().item(), "classes hit", (bc > 0).sum().item(), "min/max count", bc.min().item(), bc.max().item(), "T2 entropy", (-(zp/2).softmax(1) * (zp/2).log_softmax(1)).sum(1).mean().item())
    print(bc.tolist())
