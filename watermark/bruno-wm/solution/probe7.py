"""What does the synthetic-trained StegaStamp detector rely on? Apply perturbations to our own embedding (on clean
challenge images) and see when it stops firing. Mean probability over the image (1 = watermark seen everywhere)."""
import io, sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from common import OUT, RELEASE
from model import Net
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
meth = "stegastamp"
nets = []
for ck in sys.argv[1:]:
    n_ = Net().cuda().eval(); n_.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)); nets.append(n_)
wm = load_watermark(meth, device="cuda"); ds = LocalisationDataset(RELEASE, "train", methods=[meth]); rng = np.random.default_rng(0)
@torch.no_grad()
def p(a):
    x = torch.from_numpy(np.clip(a, 0, 255).astype(np.uint8)).permute(2, 0, 1)[None].cuda()
    with torch.autocast("cuda", dtype=torch.float16): return [float(torch.sigmoid(n_(x).float()).mean()) for n_ in nets]
def gblur(a, s):
    r = int(3 * s + 0.5); t = torch.arange(-r, r + 1, dtype=torch.float32); k = torch.exp(-t ** 2 / (2 * s * s)); k /= k.sum()
    x = torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1)[None]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)
    return x[0].permute(1, 2, 0).numpy()
def jpeg(a, q):
    b = io.BytesIO(); Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(b, format="JPEG", quality=q); return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB")).astype(np.float32)
def rs(a, s, k=Image.Resampling.BICUBIC):
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)); return np.asarray(im.resize((s, s), k).resize((512, 512), k)).astype(np.float32)
R = {}
def add(k, v): R.setdefault(k, []).append(v)
n = 0
for i in range(len(ds)):
    it = ds[i]
    if it["full_watermark"] or it["mask"].mean() > 0.05: continue
    c = it["image"].astype(np.float32); msg = rng.integers(0, 2, wm.nbits); w = wm.embed(it["image"], msg).astype(np.float32); r = w - c; n += 1
    add("clean image", p(c)); add("our embedding (reference)", p(w))
    add("residual x0.7", p(c + 0.7 * r)); add("residual x0.5", p(c + 0.5 * r)); add("residual x0.3", p(c + 0.3 * r))
    add("residual blurred s=0.7", p(c + gblur(r, 0.7))); add("residual x0.75 blurred s=0.7", p(c + 0.75 * gblur(r, 0.7))); add("residual x0.75 blurred s=1", p(c + 0.75 * gblur(r, 1.0)))
    add("residual blurred s=0.5", p(c + gblur(r, 0.5))); add("residual blurred s=1", p(c + gblur(r, 1.0))); add("residual blurred s=2", p(c + gblur(r, 2.0)))
    add("image blurred s=0.5", p(gblur(w, 0.5))); add("noise +-1", p(w + rng.integers(-1, 2, w.shape))); add("noise sd 2", p(w + rng.normal(0, 2, w.shape)))
    add("jpeg 95", p(jpeg(w, 95))); add("jpeg 85", p(jpeg(w, 85))); add("resize 512>448>512", p(rs(w, 448))); add("resize 512>1024>512", p(rs(w, 1024)))
    add("shift residual 1px", p(c + np.roll(r, 1, axis=1))); add("shift residual 4px", p(c + np.roll(r, 4, axis=1))); add("flip residual", p(c + r[:, ::-1]))
    c2 = gblur(c, 0.7); r2 = wm.embed(np.clip(c2, 0, 255).astype(np.uint8), msg).astype(np.float32) - np.clip(c2, 0, 255).astype(np.uint8)
    add("residual computed on a blurred copy, added to c", p(c + r2))
    up = np.asarray(Image.fromarray(it["image"]).resize((1024, 1024), Image.Resampling.BICUBIC)); wu = wm.embed(up, msg)
    add("embed at 1024 then lanczos down", p(np.asarray(Image.fromarray(wu).resize((512, 512), Image.Resampling.LANCZOS)).astype(np.float32)))
    add("embed at 1024 then bilinear down", p(np.asarray(Image.fromarray(wu).resize((512, 512), Image.Resampling.BILINEAR)).astype(np.float32)))
    if n >= 12: break
print(f"{'perturbation':50s} " + "  ".join(f"det{i}" for i in range(len(nets))) + "   (det0 = synthetic-trained, det1 = fine-tuned on real images)")
for k, v in R.items(): print(f"{k:50s} " + "  ".join("%.3f" % x for x in np.mean(v, 0)))
# the real watermark itself, for reference
rf = [p(ds[i]["image"].astype(np.float32)) for i in range(len(ds)) if ds.meta[i]["full_watermark"]][:20]
print(f"{'REAL full-watermark challenge images':50s} " + "  ".join("%.3f" % x for x in np.mean(rf, 0)))
