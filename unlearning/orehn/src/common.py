import os, sys, math, torch, torch.nn.functional as F
ROOT = "/p/scratch/training2666/cochod2/euroai-paris/unlearning"
os.chdir(ROOT); sys.path.insert(0, ROOT)
import task_template as kit
C = ROOT + "/orehn/cache"

def affine(x, scale, tx, ty, rot, flip):
    """per-sample crop(scale<1 zooms in)/shift/rotation/flip via affine_grid. all args tensors [B]."""
    B = len(x); cos, sin = torch.cos(rot) * scale, torch.sin(rot) * scale
    th = torch.zeros(B, 2, 3, device=x.device)
    th[:, 0, 0] = cos * flip; th[:, 0, 1] = -sin; th[:, 0, 2] = tx
    th[:, 1, 0] = sin * flip; th[:, 1, 1] = cos; th[:, 1, 2] = ty
    g = F.affine_grid(th, x.shape, align_corners=False)
    return F.grid_sample(x, g, mode="bilinear", padding_mode="reflection", align_corners=False)

def blur(x, s):
    return F.interpolate(F.interpolate(x, size=s, mode="bilinear", antialias=True), size=224, mode="bilinear")

def perturb(x, kind, g):
    """strong perturbations used to read the original's 'non-memorised' behaviour"""
    B, dev = len(x), x.device
    U = lambda a, b: torch.rand(B, generator=g, device=dev) * (b - a) + a
    one = torch.ones(B, device=dev); zero = torch.zeros(B, device=dev)
    if kind == "noise25": return x + 0.25 * torch.randn(x.shape, generator=g, device=dev)
    if kind == "noise35": return x + 0.35 * torch.randn(x.shape, generator=g, device=dev)
    if kind == "lowc_up": return x * 0.7 + 0.15
    if kind == "lowc_dn": return x * 0.7 - 0.15
    if kind == "highc": return (x * 1.35).clamp(-1, 1)
    if kind == "desat": return 0.3 * x + 0.7 * x.mean(1, keepdim=True)
    if kind == "rrc":
        s = U(0.55, 0.75); return affine(x, s, U(-1, 1) * (1 - s), U(-1, 1) * (1 - s), zero, one)
    if kind == "rot": return affine(x, one * 0.85, zero, zero, U(0.2, 0.4) * torch.sign(U(-1, 1)), one)
    if kind == "blur": return blur(x, 64)
    if kind == "cutout":
        m = (torch.rand(B, 1, 14, 14, generator=g, device=dev) > 0.4).float()
        return x * F.interpolate(m, size=224, mode="nearest")
    raise ValueError(kind)
PERTS = ["noise25", "noise35", "lowc_up", "lowc_dn", "highc", "desat", "rrc", "rrc", "rot", "rot", "blur", "cutout", "cutout"]

def train_aug(x, g, strength=1.0):
    """mild random views of forget images (stand-in for the hidden altered copies)"""
    B, dev = len(x), x.device
    U = lambda a, b: torch.rand(B, generator=g, device=dev) * (b - a) + a
    clean = (U(0, 1) < 0.25).float()
    s = 1 - U(0, 0.3) * strength; flip = torch.where(U(0, 1) < 0.5, -1.0, 1.0)
    rot = U(-0.2, 0.2) * strength * (U(0, 1) < 0.3).float()
    y = affine(x, s, U(-1, 1) * (1 - s), U(-1, 1) * (1 - s), rot, flip)
    c = (1 + U(-0.25, 0.25) * strength).view(B, 1, 1, 1); b = (U(-0.15, 0.15) * strength).view(B, 1, 1, 1)
    y = y * c + b
    sat = (1 - U(0, 0.6) * (U(0, 1) < 0.3).float() * strength).view(B, 1, 1, 1)
    y = sat * y + (1 - sat) * y.mean(1, keepdim=True)
    y = y + (U(0, 0.15) * (U(0, 1) < 0.5).float() * strength).view(B, 1, 1, 1) * torch.randn(x.shape, generator=g, device=dev)
    bl = (U(0, 1) < 0.15 * strength).view(B, 1, 1, 1).float()
    y = bl * blur(y, 112) + (1 - bl) * y
    y = y.clamp(-1, 1)
    cl = clean.view(B, 1, 1, 1)
    return cl * x + (1 - cl) * y

def make_targets(T, A, alpha, wpert=0.5, wprobe=0.25, rpow=1.0, verbose=True):
    """T2 target q_i = pi_i e_y + (1-pi_i) r_i ; pi_i = sigmoid(alpha*s_i + beta), mean(pi)=A."""
    y = T["y"]; N = len(y); idx = torch.arange(N)
    def mask(P):
        P = P.clone(); P[idx, y] = 0; return P / P.sum(1, keepdim=True).clamp_min(1e-8)
    rp = mask(T["R"]); rb = mask(T["P9"])
    rc = torch.zeros(100, 100).index_add_(0, y, rp) / torch.bincount(y, minlength=100).clamp_min(1).unsqueeze(1)
    r = wpert * rp + wprobe * rb + (1 - wpert - wprobe) * rc[y]
    r = mask(r.pow(rpow))
    s = T["s"]
    lo, hi = -30.0, 30.0
    for _ in range(60):
        beta = (lo + hi) / 2
        if torch.sigmoid(alpha * s + beta).mean() < A: lo = beta
        else: hi = beta
    pi = torch.sigmoid(alpha * s + beta).clamp(1e-3, 0.995)
    q = (1 - pi).unsqueeze(1) * r; q[idx, y] = pi
    p1 = q.pow(2); p1 = p1 / p1.sum(1, keepdim=True); L = -p1[idx, y].log()
    if verbose:
        print(f"targets A={A} alpha={alpha}: pi q10/50/90={pi.quantile(torch.tensor([.1,.5,.9])).tolist()} T2-argmax acc={(q.argmax(1)==y).float().mean():.3f} "
              f"implied T1 loss mean={L.mean():.3f} std={L.std():.3f} mass on absent classes={q[:, torch.bincount(y, minlength=100)==0].sum(1).mean():.3f}", flush=True)
    return q
