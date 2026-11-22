"""Dump logits of submitted / candidate checkpoints on forget views + proxies (for calibration against official scores)."""
import glob, sys
from common import *
import numpy as np
imgs = kit.load_forget_images(); N = len(imgs)
prox = torch.from_numpy(np.load(ROOT + "/orehn/data/aigen16k.npz")["images"][14000:16000])
V = {"clean": None, "flip": lambda x: x.flip(-1), "crop": lambda x: F.interpolate(x[..., 13:-13, 13:-13], size=224, mode="bilinear"),
     "bright": lambda x: x * 0.8 + 0.1, "noise": lambda x: x + 0.2 * torch.randn(x.shape, device=x.device, generator=torch.Generator(device=x.device).manual_seed(5)),
     "shift": lambda x: torch.roll(x, (12, -9), (2, 3)), "blur": lambda x: blur(x, 160)}
@torch.no_grad()
def run(m, images, fn):
    out = []
    for i in range(0, len(images), 125):
        x = kit.preprocess(images[i:i+125]); x = fn(x) if fn else x
        with torch.autocast("cuda", dtype=torch.float16): out.append(m(x).float().cpu())
    return torch.cat(out).half()
paths = {"orig": ROOT + "/model.pt"}
for p in sorted(glob.glob(ROOT + "/orehn/ckpt/*.pt")): paths[os.path.basename(p)[:-3]] = p
for p in sys.argv[1:]: paths[os.path.basename(p)[:-3]] = p
out = f"{C}/dumps"; os.makedirs(out, exist_ok=True)
for name, p in paths.items():
    if os.path.exists(f"{out}/{name}.pt"): continue
    try:
        m = kit.build_model(); m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True), strict=True); m = m.cuda().eval()
    except Exception as e:
        print("skip", name, repr(e)[:100]); continue
    d = {k: run(m, imgs, fn) for k, fn in V.items()}; d["proxy"] = run(m, prox, None)
    torch.save(d, f"{out}/{name}.pt"); print("dumped", name, flush=True)
