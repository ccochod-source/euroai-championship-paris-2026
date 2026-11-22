"""Helpers for the occlusion / random-mask experiments (s3/occl). See s3/README.md and s3/common.py."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

from common import *  # noqa: F401,F403

DEV = "cuda"
import builtins as _bi
_LOGF = [None]


def set_log(tag):
    """Also append every print to occl/logs/<tag>.log (run.sh pipes stdout through a block-buffered grep)."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(d, exist_ok=True)
    _LOGF[0] = open(os.path.join(d, tag + ".log"), "a")


def print(*a, **k):
    _bi.print(*a, **k)
    if _LOGF[0] is not None:
        _LOGF[0].write(" ".join(str(x) for x in a) + "\n"); _LOGF[0].flush()


# ----------------------------------------------------------------------------------------------- decoder queries
@torch.no_grad()
def slog(wm, imgs, sm, bs=128):
    """Signed per-bit logits (logit * sign of the known bit), B x nbits float32 on the GPU."""
    out = []
    for i in range(0, len(imgs), bs):
        x = imgs[i:i + bs]
        if wm.name == "stegastamp":
            lg = wm.logits(x.clamp(0, 1), bs=32).to(sm.device)
        elif wm.name == "trustmark":
            with torch.autocast("cuda", dtype=torch.float16):
                lg = wm.logits(x)
            lg = lg.float()
        else:
            lg = wm.logits(x).float()
        out.append(lg * sm[None])
    return torch.cat(out)


def slog_grad(wm, imgs, sm):
    """Differentiable signed logits (trustmark / pixelseal only), float32."""
    if wm.name == "trustmark":
        lg = wm.net(wm._prep(imgs)).float()
    elif wm.name == "pixelseal":
        lg = wm.det(wm._resize(imgs))[:, 1:].float()
    else:
        raise ValueError(wm.name)
    return lg * sm[None]


# ----------------------------------------------------------------------------------------------- fills
def gauss_blur(img, sigma):
    k = int(2 * round(3 * sigma) + 1)
    x = torch.arange(k, device=img.device, dtype=torch.float32) - k // 2
    w = torch.exp(-x ** 2 / (2 * sigma ** 2)); w /= w.sum()
    c = img.shape[1]
    y = F.pad(img, (k // 2,) * 4, mode="reflect")
    y = F.conv2d(y, w.view(1, 1, 1, k).expand(c, 1, 1, k), groups=c)
    return F.conv2d(y, w.view(1, 1, k, 1).expand(c, 1, k, 1), groups=c)


def make_fill(img, kind, other=None):
    """img 1x3xHxW. kind: grey | mean | blurS (S = sigma in px) | downS (down/up-sample by S) | other | shiftS | flip"""
    if kind == "grey":
        return torch.full_like(img, 0.5)
    if kind == "mean":
        return img.mean((2, 3), keepdim=True).expand_as(img).contiguous()
    if kind.startswith("blur"):
        return gauss_blur(img, float(kind[4:]))
    if kind.startswith("down"):
        s = int(kind[4:]); h = img.shape[-1]
        y = F.interpolate(img, size=(h // s, h // s), mode="bilinear", antialias=True, align_corners=False)
        return F.interpolate(y, size=(h, h), mode="bicubic", align_corners=False).clamp(0, 1)
    if kind.startswith("shift"):
        s = int(kind[5:]); return torch.roll(img, (s, s), (2, 3))
    if kind == "flip":
        return img.flip(3)
    if kind == "other":
        return other
    if kind.startswith("noise"):  # blur + gaussian noise
        return (gauss_blur(img, 8.0) + float(kind[5:]) * torch.randn_like(img)).clamp(0, 1)
    raise ValueError(kind)


# ----------------------------------------------------------------------------------------------- random masks
def rand_masks(K, g, p, soft, seed=0, shift=False):
    """K random keep-masks. Returns cells (K x g*g, 1 = kept) and masks K x 1 x 512 x 512 in [0,1].
    soft: bilinear upsampling of the cell grid (RISE) instead of nearest. shift: random translation of the grid
    (then `cells` is only approximate; use the 128x128 pooled masks as design matrix)."""
    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    if not shift:
        cells = (torch.rand(K, 1, g, g, device=DEV, generator=gen) < p).float()
        m = F.interpolate(cells, size=(512, 512), mode="bilinear" if soft else "nearest",
                          **(dict(align_corners=False) if soft else {}))
        return cells.view(K, g * g), m
    c = 512 // g
    cells = (torch.rand(K, 1, g + 1, g + 1, device=DEV, generator=gen) < p).float()
    big = F.interpolate(cells, size=((g + 1) * c,) * 2, mode="bilinear" if soft else "nearest",
                        **(dict(align_corners=False) if soft else {}))
    ox = torch.randint(0, c, (K,), generator=gen, device=DEV); oy = torch.randint(0, c, (K,), generator=gen, device=DEV)
    m = torch.stack([big[k, :, oy[k]:oy[k] + 512, ox[k]:ox[k] + 512] for k in range(K)])
    return None, m


def ridge(X, Y, lam):
    """X: K x P (0/1 keep), Y: K x D. Returns weights P x D of the centred ridge regression Y ~ b + X w."""
    Xc = X - X.mean(0, keepdim=True); Yc = Y - Y.mean(0, keepdim=True)
    A = Xc.T @ Xc
    A = A + lam * A.diagonal().mean() * torch.eye(A.shape[0], device=X.device)
    return torch.linalg.solve(A.double(), (Xc.T @ Yc).double()).float()


def ridge_dual(X, Y, lam):
    """Same as ridge but in the dual (K x K system), for P >> K. X: K x P, Y: K x D -> P x D."""
    Xc = X - X.mean(0, keepdim=True); Yc = Y - Y.mean(0, keepdim=True)
    G = Xc @ Xc.T
    G = G + lam * G.diagonal().mean() * torch.eye(G.shape[0], device=X.device)
    return (Xc.T.double() @ torch.linalg.solve(G.double(), Yc.double())).float()


@torch.no_grad()
def query_masks(wm, img, fill, masks, sm, bs=128):
    """Signed logits K x nbits for composites mask*img + (1-mask)*fill."""
    out = []
    for i in range(0, len(masks), bs):
        m = masks[i:i + bs]
        out.append(slog(wm, m * img + (1 - m) * fill, sm, bs=bs))
    return torch.cat(out)


# ----------------------------------------------------------------------------------------------- evaluation
def to128(ev, mode="bilinear", smooth=0.0):
    """ev: N x h x w torch / numpy -> N x 128 x 128 numpy float32."""
    t = torch.as_tensor(np.asarray(ev), dtype=torch.float32)[:, None]
    if t.shape[-1] != 128:
        t = F.interpolate(t, size=(128, 128), mode=mode, **(dict(align_corners=False) if mode != "nearest" else {}))
    if smooth > 0:
        t = gauss_blur(t, smooth)
    return t[:, 0].numpy()


def auc_img(ev, gt):
    """AUC of 'watermarked cells have higher evidence than hole cells'. nan if one class is empty."""
    pos = ev[gt]; neg = ev[~gt]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


QS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40]
TS = [-1.0, -0.5, -0.25, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ZS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
BINS = [(0, .02, "<2%"), (.02, .06, "2-6%"), (.06, .12, "6-12%"), (.12, 1.01, ">12%")]


def rule_preds(ev):
    """Candidate binary predictions (1 = watermarked) of one evidence map, for the three families of rules that do
    not look at the ground truth: quantile q, fraction t of the median evidence, robust z-score."""
    out = {}
    srt = np.sort(ev.ravel())
    for q in QS:
        out[("q", q)] = ev >= srt[int(q * (ev.size - 1))] if q > 0 else np.ones_like(ev, bool)
    med = np.median(ev)
    for t in TS:
        out[("t", t)] = ev >= t * med
    mad = np.median(np.abs(ev - med)) * 1.4826 + 1e-12
    for z in ZS:
        out[("z", z)] = ev >= med - z * mad
    return out


def evaluate(maps128, names, G, label="", folds=5, verbose=True, seed=0):
    """maps128: N x 128 x 128. Returns dict with mean AUC, all-ones score, cross-validated mIoU of the best rule of
    each family (rule parameter chosen on the other folds), oracle-threshold mIoU, and per hole-size-bin scores."""
    N = len(names)
    gts = [G[n] for n in names]
    aucs = np.array([auc_img(maps128[i], gts[i]) for i in range(N)])
    ones = np.array([miou(np.ones((128, 128)), g) for g in gts])
    hole = np.array([1 - g.mean() for g in gts])
    keys = None; S = []
    for i in range(N):
        pr = rule_preds(maps128[i])
        if keys is None:
            keys = list(pr)
        S.append([miou(pr[k], gts[i]) for k in keys])
    S = np.array(S)  # N x rules
    rng = np.random.RandomState(seed); perm = rng.permutation(N); fold = np.empty(N, int); fold[perm] = np.arange(N) % folds
    res = dict(auc=float(np.nanmean(aucs)), ones=float(ones.mean()), n=N)
    cvs = {}
    for fam in ["q", "t", "z", "any"]:
        idx = [j for j, k in enumerate(keys) if fam == "any" or k[0] == fam]
        cv = np.zeros(N); chosen = []
        for f in range(folds):
            tr = fold != f
            j = idx[int(np.argmax(S[tr][:, idx].mean(0)))]
            cv[~tr] = S[~tr, j]; chosen.append(keys[j])
        cvs[fam] = cv; res["cv_" + fam] = float(cv.mean()); res["par_" + fam] = max(set(chosen), key=chosen.count)
    # oracle: best threshold per image over the quantile family (upper bound of the map, NOT a legal score)
    res["oracle"] = float(S.max(1).mean())
    best = max(["q", "t", "z"], key=lambda f: res["cv_" + f])
    res["best_fam"] = best
    res["bins"] = {lab: (float(cvs[best][(hole >= a) & (hole < b)].mean()) if ((hole >= a) & (hole < b)).any() else float("nan"),
                         float(ones[(hole >= a) & (hole < b)].mean()) if ((hole >= a) & (hole < b)).any() else float("nan"),
                         float(np.nanmean(aucs[(hole >= a) & (hole < b)])) if ((hole >= a) & (hole < b)).any() else float("nan"),
                         int(((hole >= a) & (hole < b)).sum())) for a, b, lab in BINS}
    if verbose:
        b = " ".join(f"{lab}:{v[0]:.3f}/{v[1]:.3f}(auc{v[2]:.2f},n{v[3]})" for lab, v in res["bins"].items())
        print(f"{label:46s} AUC {res['auc']:.3f} | cv q {res['cv_q']:.3f}{res['par_q']} t {res['cv_t']:.3f}{res['par_t']} "
              f"z {res['cv_z']:.3f}{res['par_z']} | ones {res['ones']:.3f} oracle {res['oracle']:.3f} | {b}", flush=True)
    return res


# ----------------------------------------------------------------------------------------------- partitions
def grid_labels(g, H=512):
    """H x H long label map of a g x g grid (row-major)."""
    i = torch.arange(H, device=DEV) * g // H
    return i[:, None] * g + i[None, :]


def slic(img, n=32, m=0.1, iters=8):
    """Minimal GPU SLIC (no connectivity enforcement). img 1x3xHxW in [0,1]. Returns H x W long labels in [0, n*n).
    m: colour difference that weighs as much as one grid step (smaller = follows colour edges more)."""
    H, W = img.shape[-2:]; S = H / n
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device), indexing="ij")
    f = torch.cat([img[0] / m, (xx[None] / S).float(), (yy[None] / S).float()])          # 5 x H x W
    C = F.adaptive_avg_pool2d(f[None], n)[0]                                             # 5 x n x n
    gi = (yy.float() / S).long().clamp(max=n - 1); gj = (xx.float() / S).long().clamp(max=n - 1)
    lab = gi * n + gj
    for _ in range(iters):
        best = torch.full((H, W), 1e9, device=img.device)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ci = gi + di; cj = gj + dj
                ok = (ci >= 0) & (ci < n) & (cj >= 0) & (cj < n)
                cc = C[:, ci.clamp(0, n - 1), cj.clamp(0, n - 1)]
                d = ((f - cc) ** 2).sum(0); d[~ok] = 1e9
                upd = d < best; best = torch.where(upd, d, best); lab = torch.where(upd, ci * n + cj, lab)
        flat = lab.reshape(-1)
        sums = torch.zeros(n * n, 5, device=img.device).index_add_(0, flat, f.reshape(5, -1).T)
        cnt = torch.zeros(n * n, device=img.device).index_add_(0, flat, torch.ones(flat.shape[0], device=img.device))
        C = torch.where(cnt.view(1, n, n) > 0, (sums / cnt.clamp(min=1)[:, None]).T.reshape(5, n, n), C)
    return lab


@torch.no_grad()
def regress_partition(wm, img, fill, sm, lab, P, K, p, active=None, seed=0, lam=0.05, bs=128):
    """Random-mask ridge regression on an arbitrary partition. lab: H x W long labels in [0, P). Every ACTIVE region
    is kept with probability p in each of the K queries (inactive regions are always kept).
    Returns (w, t): per-region weight of the margin (mean over bits) and its t-statistic over bits, length P
    (0 for inactive regions), plus the pixel count of each region."""
    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    keep = (torch.rand(K, P, device=DEV, generator=gen) < p).float()
    if active is not None:
        keep[:, ~active] = 1
    Y = []
    for kb in keep.split(bs):
        m = kb[:, lab][:, None]
        Y.append(slog(wm, m * img + (1 - m) * fill, sm, bs=bs))
    Y = torch.cat(Y)
    idx = torch.arange(P, device=DEV) if active is None else active.nonzero()[:, 0]
    Wb = ridge(keep[:, idx], Y, lam)
    w = torch.zeros(P, device=DEV); t = torch.zeros(P, device=DEV)
    w[idx] = Wb.mean(1); t[idx] = Wb.mean(1) / (Wb.std(1) + 1e-6)
    area = torch.bincount(lab.reshape(-1), minlength=P).float()
    return w, t, area
