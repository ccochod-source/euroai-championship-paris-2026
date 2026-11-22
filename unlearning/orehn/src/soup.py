"""50/50 weight average of two checkpoints + the usual local metrics."""
import sys, json, numpy as np
from common import *
a_, b_, out = sys.argv[1], sys.argv[2], sys.argv[3]
sa = torch.load(a_, map_location="cpu", weights_only=True); sb = torch.load(b_, map_location="cpu", weights_only=True)
sd = {k: ((sa[k].float() + sb[k].float()) / 2) if sa[k].is_floating_point() else sa[k] for k in sa}
assert all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point())
torch.save(sd, out + ".tmp"); os.replace(out + ".tmp", out)
m = kit.build_model(); m.load_state_dict(torch.load(out, map_location="cpu", weights_only=True), strict=True); m = m.cuda().eval(); orig = kit.load_model()
T = torch.load(C + "/teacher.pt"); y = T["y"].cuda(); imgs = kit.load_forget_images()
prox = torch.from_numpy(np.load(ROOT + "/orehn/data/aigen16k.npz")["images"][14000:16000]); zo = torch.load(C + "/proxy_logits.pt")[14000:16000].cuda()
@torch.no_grad()
def lg(mm, images, fn=None):
    o = []
    for i in range(0, len(images), 125):
        x = kit.preprocess(images[i:i+125]); x = fn(x) if fn else x
        with torch.autocast("cuda", dtype=torch.float16): o.append(mm(x).float())
    return torch.cat(o)
V = {"clean": None, "crop": lambda x: F.interpolate(x[..., 13:-13, 13:-13], size=224, mode="bilinear"), "noise.2": lambda x: x + 0.2 * torch.randn(x.shape, device=x.device, generator=torch.Generator(device=x.device).manual_seed(5))}
res = {}
for k, fn in V.items():
    z = lg(m, imgs, fn); L = F.cross_entropy(z, y, reduction="none"); res[k] = (round((z.argmax(1) == y).float().mean().item(), 4), round(L.mean().item(), 3), round(L.std().item(), 3))
z = lg(m, prox); kl = ((zo/2).softmax(1) * ((zo/2).log_softmax(1) - (z/2).log_softmax(1))).sum(1).mean().item(); ag = (z.argmax(1) == zo.argmax(1)).float().mean().item()
res["proxy"] = dict(KL=round(kl, 4), agree=round(ag, 4), retain_est=round(ag / (1 + kl), 4))
print(json.dumps(res), flush=True); print("saved", out, os.path.getsize(out))
