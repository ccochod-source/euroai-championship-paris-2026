"""TrustMark: use its own watermark REMOVER as the 'fill' (same content, watermark removed) so that occlusion does
not change the picture content. Sanity numbers on real full images, then the expected mask-gradient with that fill."""
import sys, time, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from common import *
from scipy.stats import rankdata

wm = load_wm("trustmark"); dev = "cuda"; K = int(sys.argv[1]) if len(sys.argv) > 1 else 32; G = 64
for p in wm.net.parameters(): p.requires_grad_(False)
def logits(x):
    r = wm.resolution
    return wm.net(F.interpolate(x, size=(r, r), mode="bilinear", align_corners=False, antialias=True) * 2 - 1).float()
def removed(path):
    out = wm.tm.remove_watermark(Image.open(path).convert("RGB"))
    return torch.from_numpy(np.asarray(out.convert("RGB"), dtype=np.uint8)).permute(2, 0, 1)[None].float().div(255).to(dev)
def auc(score, pos):
    r = rankdata(score); n1 = pos.sum(); n0 = (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
def smooth(a, s):
    t = torch.from_numpy(a)[None, None].float(); k = int(4 * s) | 1; ax = torch.arange(k) - k // 2; g = torch.exp(-ax ** 2 / (2 * s * s)); g = g / g.sum()
    t = F.conv2d(F.pad(t, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k)); t = F.conv2d(F.pad(t, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1))
    return t[0, 0].numpy()

g = gt128(); items = [it for it in load_items("train", "trustmark") if it["full"]]
S = []; maps = {"eg": [], "absres": [], "gxr": []}; names = []; t0 = time.time()
for it in items:
    x = load_img(it["path"]); sm = sm_of(it["bits"]); xr = removed(it["path"])
    m = torch.from_numpy(gt512(it["name"])).to(dev)[None, None].float()
    with torch.no_grad():
        v = torch.cat([x, xr, x * m + xr * (1 - m), xr * m + x * (1 - m)])
        mg = (logits(v) * sm[None]).mean(1).cpu().numpy()
    S.append(np.concatenate([mg, [float(((x - xr) ** 2).mean().sqrt() * 255)]]))
    # (a) expected mask-gradient with the remover output as fill
    acc = torch.zeros(G, G, device=dev)
    for _ in range(K // 16):
        ms = [F.interpolate((torch.rand(1, 1, g0, g0, device=dev) < p).float(), size=(G, G), mode="bilinear", align_corners=False)
              for g0, p in [(int(np.random.choice([4, 8, 16])), float(np.random.uniform(0.6, 0.97))) for _ in range(16)]]
        mm = torch.cat(ms).requires_grad_(True); M = F.interpolate(mm, size=(512, 512), mode="bilinear", align_corners=False)
        acc += torch.autograd.grad((logits(x * M + xr * (1 - M)) * sm[None]).mean(1).sum(), mm)[0].sum((0, 1))
    maps["eg"].append(F.interpolate(acc[None, None], size=(128, 128), mode="bilinear", align_corners=False)[0, 0].cpu().numpy())
    # (b) magnitude of what the remover removed, per cell  (c) plain gradient x (x - removed) at the image itself
    maps["absres"].append(F.avg_pool2d((x - xr).abs().mean(1, keepdim=True), 4)[0, 0].cpu().numpy())
    xg = x.clone().requires_grad_(True); gr = torch.autograd.grad((logits(xg) * sm[None]).mean(1).sum(), xg)[0]
    maps["gxr"].append(F.avg_pool2d((gr * (x - xr)).sum(1, keepdim=True), 4)[0, 0].cpu().numpy() * 16)
    names.append(it["name"])
S = np.array(S); G8 = np.array([g[n] for n in names]); hole = 1 - G8.reshape(len(names), -1).mean(1)
print(f"n={len(names)} {(time.time()-t0)/len(names):.2f}s/img | margin: image {S[:,0].mean():.2f} | fully removed {S[:,1].mean():.2f} | holes<-removed {S[:,2].mean():.2f} | watermarked<-removed {S[:,3].mean():.2f} | RMS(x - removed) {S[:,4].mean():.2f} grey levels")
for k, A0 in maps.items():
    np.savez_compressed(f"{OUT}/rm_{k}_trustmark_train.npz", names=np.array(names), maps=np.array(A0).astype(np.float32))
    for s in [0, 1.5, 3]:
        A = [smooth(a, s) if s else a for a in A0]
        aucs = np.array([auc(a.ravel(), x.ravel()) for a, x in zip(A, G8)])
        line = f"{k:7s} smooth {s}: AUC mean {np.nanmean(aucs):.3f} median {np.nanmedian(aucs):.3f} | mIoU"
        for tau in [0.15, 0.3, 0.5, 0.7]:
            line += f" tau{tau}: {np.mean([miou(a > tau * np.median(a), x) for a, x in zip(A, G8)]):.4f}"
        print(line + " | AUC by hole size " + " ".join(f"{np.nanmean(aucs[(hole >= lo) & (hole < hi)]):.2f}" for lo, hi in [(0, .02), (.02, .06), (.06, .12), (.12, 1)]))
