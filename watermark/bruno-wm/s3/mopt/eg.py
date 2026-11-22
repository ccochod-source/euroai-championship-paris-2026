"""Expected mask-gradient evidence for holes in whole-frame images (trustmark, pixelseal: differentiable decoders).

For K random keep-masks M_k (coarse random cells, mostly kept), the image is x*M + fill*(1-M); we take the gradient of
the decoder margin with respect to a fine mask (G x G cells) at M_k and average over k. A cell that carries the
watermark has a positive gradient (removing it lowers the margin), a hole has ~0.
usage: eg.py <method> [K] [G] [fill] [limit]
"""
import sys, time, numpy as np, torch, torch.nn.functional as F
from common import *

meth = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 32; G = int(sys.argv[3]) if len(sys.argv) > 3 else 64
fill_kind = sys.argv[4] if len(sys.argv) > 4 else "grey"; limit = int(sys.argv[5]) if len(sys.argv) > 5 else 0
wm = load_wm(meth); dev = "cuda"
for p in (wm.net if meth == "trustmark" else wm.det).parameters():
    p.requires_grad_(False)


def logits(x):  # x: B x 3 x 512 x 512 in [0,1], differentiable
    if meth == "trustmark":
        r = wm.resolution
        return wm.net(F.interpolate(x, size=(r, r), mode="bilinear", align_corners=False, antialias=True) * 2 - 1).float()
    return wm.det(F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=True))[:, 1:].float()


def blur(x, s):
    k = int(4 * s) | 1; ax = torch.arange(k, device=x.device) - k // 2; g = torch.exp(-ax ** 2 / (2 * s * s)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


def evidence(img, sm, K=K, G=G, bs=16, gen=None):
    fill = torch.full_like(img, 0.5) if fill_kind == "grey" else (blur(img, 12.0) if fill_kind == "blur" else img.mean((2, 3), keepdim=True).expand_as(img))
    acc = torch.zeros(G, G, device=dev); done = 0
    while done < K:
        b = min(bs, K - done); done += b
        ms = []
        for _ in range(b):
            g0 = int(np.random.choice([4, 8, 16])); p = float(np.random.uniform(0.7, 0.97))
            ms.append(F.interpolate((torch.rand(1, 1, g0, g0, device=dev) < p).float(), size=(G, G), mode="bilinear", align_corners=False))
        m = torch.cat(ms).requires_grad_(True)
        M = F.interpolate(m, size=img.shape[-2:], mode="bilinear", align_corners=False)
        S = (logits(img * M + fill * (1 - M)) * sm[None]).mean(1).sum()
        acc += torch.autograd.grad(S, m)[0].sum((0, 1))
    return (acc / K).detach()


def smooth(a, s):
    if s <= 0: return a
    t = torch.from_numpy(a)[None, None].float(); k = int(4 * s) | 1; ax = torch.arange(k) - k // 2; g = torch.exp(-ax ** 2 / (2 * s * s)); g = g / g.sum()
    t = F.conv2d(F.pad(t, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k)); t = F.conv2d(F.pad(t, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1))
    return t[0, 0].numpy()


def auc(score, pos):
    from scipy.stats import rankdata
    r = rankdata(score); n1 = pos.sum(); n0 = (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


g = gt128(); items = [it for it in load_items("train", meth) if it["full"]]
if limit: items = items[:limit]
maps, names, t0 = [], [], time.time()
for it in items:
    e = evidence(load_img(it["path"]), sm_of(it["bits"]))
    e128 = F.interpolate(e[None, None], size=(128, 128), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    maps.append(e128); names.append(it["name"])
dt = (time.time() - t0) / len(items)
maps = np.array(maps); np.savez_compressed(f"{OUT}/eg_{meth}_train_K{K}_G{G}_{fill_kind}.npz", names=np.array(names), maps=maps.astype(np.float32))
G8 = np.array([g[n] for n in names]); hole = 1 - G8.reshape(len(names), -1).mean(1)
print(f"== {meth} K={K} G={G} fill={fill_kind} n={len(names)} {dt:.2f}s/img  all-ones mIoU {np.mean([miou(np.ones((128,128)), x) for x in G8]):.4f}")
for s in [0, 1.5, 3]:
    A = np.array([smooth(m, s) for m in maps])
    aucs = [auc(a.ravel(), x.ravel()) for a, x in zip(A, G8)]
    line = f"   smooth {s}: AUC mean {np.nanmean(aucs):.3f} median {np.nanmedian(aucs):.3f} |"
    for tau in [0.15, 0.25, 0.35, 0.5]:
        sc = [miou(a > tau * np.median(a), x) for a, x in zip(A, G8)]
        line += f" tau {tau}: {np.mean(sc):.4f}"
    print(line)
    if s == 1.5:
        sc = np.array([miou(a > 0.25 * np.median(a), x) for a, x in zip(A, G8)])
        for lo, hi in [(0, .02), (.02, .06), (.06, .12), (.12, 1)]:
            k = (hole >= lo) & (hole < hi); print(f"      hole {lo:.2f}-{hi:.2f}: n={k.sum()} mIoU {sc[k].mean():.3f} AUC {np.nanmean(np.array(aucs)[k]):.3f}")
