"""Diagnostic: original model on forget images + views; save logits and per-block CLS features."""
import sys, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/p/scratch/training2666/cochod2/euroai-paris/unlearning")
import os; os.chdir("/p/scratch/training2666/cochod2/euroai-paris/unlearning")
import task_template as kit
OUT = "orehn/cache"; os.makedirs(OUT, exist_ok=True)
model = kit.load_model(); imgs = kit.load_forget_images()
print("images", imgs.shape, flush=True)
def views(x, kind, g):
    if kind == "clean": return x
    if kind == "flip": return x.flip(-1)
    if kind == "crop": return F.interpolate(x[..., 13:-13, 13:-13], size=224, mode="bilinear", align_corners=False)
    if kind == "noise": return x + 0.1 * torch.randn(x.shape, generator=g, device=x.device)
    if kind == "noise3": return x + 0.3 * torch.randn(x.shape, generator=g, device=x.device)
    if kind == "blur":
        return F.interpolate(F.interpolate(x, size=112, mode="bilinear", antialias=True), size=224, mode="bilinear")
    if kind == "bright": return (x * 0.8 + 0.1).clamp(-1, 1)
    if kind == "gray": return x.mean(1, keepdim=True).expand_as(x)
    if kind == "rot": return torch.rot90(x, 1, (2, 3))
@torch.no_grad()
def run(kind, feats=False):
    g = torch.Generator(device="cuda").manual_seed(0); L = []; Fs = []
    for i in range(0, len(imgs), 125):
        x = views(kit.preprocess(imgs[i:i+125]), kind, g)
        with torch.autocast("cuda", dtype=torch.float16):
            h = model.patch_embed(x); h = model._pos_embed(h); h = model.norm_pre(h); cl = []
            for b in model.blocks:
                h = b(h); cl.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu())
            z = model.forward_head(model.norm(h))
        L.append(z.float().cpu())
        if feats: Fs.append(torch.stack(cl, 1).half())
    return torch.cat(L), (torch.cat(Fs) if feats else None)
res = {}
z0, f0 = run("clean", True); y = z0.argmax(1); res["clean"] = z0
torch.save(f0, f"{OUT}/blockfeats_clean.pt")
print("class counts min/max/nonzero", torch.bincount(y, minlength=100).min().item(), torch.bincount(y, minlength=100).max().item(), (torch.bincount(y, minlength=100) > 0).sum().item())
print("bincount", torch.bincount(y, minlength=100).tolist())
for k in ["clean", "flip", "crop", "noise", "noise3", "blur", "bright", "gray", "rot"]:
    z = z0 if k == "clean" else run(k)[0]; res[k] = z
    p = z.softmax(1); conf = p.max(1).values; ce = F.cross_entropy(z, y, reduction="none")
    p2 = (z / 2).softmax(1)
    q = torch.tensor([.01, .05, .1, .25, .5, .75, .9])
    print(f"{k:7s} agree={(z.argmax(1)==y).float().mean():.4f} conf mean={conf.mean():.4f} q={[round(v,3) for v in conf.quantile(q).tolist()]} ce mean={ce.mean():.4f} std={ce.std():.4f} T2conf={p2.max(1).values.mean():.4f} logitmax={z.max(1).values.mean():.2f} top2gap={(z.topk(2).values[:,0]-z.topk(2).values[:,1]).mean():.2f}", flush=True)
torch.save(res, f"{OUT}/orig_logits_views.pt")
sd = torch.load("model.pt", map_location="cpu", weights_only=True)
for k, v in sd.items():
    if "ls" in k or "gamma" in k: print(k, v.abs().mean().item(), v.abs().max().item())
print("head norm", sd["head.weight"].norm(dim=1).mean().item(), "bias", sd["head.bias"].abs().mean().item())
# contact sheet
from PIL import Image
order = torch.argsort(y, stable=True)
for part in range(4):
    sheet = Image.new("RGB", (112*10, 112*25))
    for r, c in enumerate(range(part*25, part*25+25)):
        idx = (y == c).nonzero().flatten()[:10]
        for j, ii in enumerate(idx.tolist()):
            sheet.paste(Image.fromarray(imgs[ii].numpy()).resize((112, 112)), (j*112, r*112))
    sheet.save(f"{OUT}/sheet{part}.jpg", quality=80)
print("done")
