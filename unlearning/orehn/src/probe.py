import torch, os, torch.nn.functional as F
os.chdir("/p/scratch/training2666/cochod2/euroai-paris/unlearning/orehn/cache")
f = torch.load("blockfeats_clean.pt").float().cuda(); R = torch.load("orig_logits_views.pt")
y = R["clean"].argmax(1).cuda(); N = len(y)
present = torch.bincount(y, minlength=100) > 0
for k in ["noise3", "bright", "gray", "rot", "blur"]:
    z = R[k].cuda(); w = z.argmax(1) != y
    print(k, "wrong", w.float().mean().item(), "wrong->absent class frac", (~present[z.argmax(1)[w]]).float().mean().item(),
          "T2 mass on absent (all imgs)", (z/2).softmax(1)[:, ~present].sum(1).mean().item())
z = R["clean"].cuda(); print("clean T1 mass absent", z.softmax(1)[:, ~present].sum(1).mean().item(), "2nd-best class absent frac", (~present[z.topk(2).indices[:,1]]).float().mean().item())
g = torch.Generator(device="cuda").manual_seed(0); perm = torch.randperm(N, generator=g, device="cuda"); fold = torch.empty(N, dtype=torch.long, device="cuda"); fold[perm] = torch.arange(N, device="cuda") % 5
Y = F.one_hot(y, 100).float()
def cv(X, lam):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6); X = torch.cat([X, torch.ones(N, 1, device="cuda")], 1); P = torch.zeros(N, 100, device="cuda")
    for k in range(5):
        tr, te = fold != k, fold == k
        A = X[tr].T @ X[tr] + lam * tr.sum() * torch.eye(X.shape[1], device="cuda")
        W = torch.linalg.solve(A, X[tr].T @ Y[tr]); P[te] = X[te] @ W
    return P
for l in range(12):
    for name, X in (("cls", f[:, l, :768]), ("mean", f[:, l, 768:]), ("both", f[:, l])):
        best = max(((cv(X, lam).argmax(1) == y).float().mean().item(), lam) for lam in (0.01, 0.1, 1.0))
        print(f"block{l} {name}: cv acc {best[0]:.4f} lam {best[1]}", flush=True)
