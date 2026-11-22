"""Which resampling history reproduces the fingerprint of the challenge images?
Fingerprints: matched-filter slopes per band (StegaStamp and TrustMark) and the responses of two StegaStamp
detectors (det0 trained on our embedding, det1 fine-tuned on challenge images)."""
import io, sys, numpy as np, torch
from PIL import Image
from common import OUT, RELEASE
from model import Net
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from bands import stat
B = {"bicubic": Image.Resampling.BICUBIC, "lanczos": Image.Resampling.LANCZOS, "bilinear": Image.Resampling.BILINEAR}
def rsz(a, s, k="bicubic"): return np.asarray(Image.fromarray(a).resize((s, s), B[k]), dtype=np.uint8)
def jpeg(a, q):
    b = io.BytesIO(); Image.fromarray(a).save(b, format="JPEG", quality=q); return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), dtype=np.uint8)
CAND = {
    "embed@512 (control)": lambda c, e: e(c),
    "embed@512, 512>448>512 bicubic": lambda c, e: rsz(rsz(e(c), 448), 512),
    "embed@512, 512>384>512 bicubic": lambda c, e: rsz(rsz(e(c), 384), 512),
    "embed@512, 512>640>512 bicubic": lambda c, e: rsz(rsz(e(c), 640), 512),
    "embed@448 then up to 512 bicubic": lambda c, e: rsz(e(rsz(c, 448)), 512),
    "embed@384 then up to 512 bicubic": lambda c, e: rsz(e(rsz(c, 384)), 512),
    "embed@256 then up to 512 bicubic": lambda c, e: rsz(e(rsz(c, 256)), 512),
    "embed@400 then up to 512 bilinear": lambda c, e: rsz(e(rsz(c, 400)), 512, "bilinear"),
    "embed@640 then down to 512 bicubic": lambda c, e: rsz(e(rsz(c, 640)), 512),
    "embed@768 then down to 512 lanczos": lambda c, e: rsz(e(rsz(c, 768)), 512, "lanczos"),
    "embed@720 then down to 512 bicubic": lambda c, e: rsz(e(rsz(c, 720)), 512),
    "embed@512, jpeg 90": lambda c, e: jpeg(e(c), 90),
    "embed@512, jpeg 75": lambda c, e: jpeg(e(c), 75),
}
nets = []
for ck in sys.argv[1:3]:
    n_ = Net().cuda().eval(); n_.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)); nets.append(n_)
@torch.no_grad()
def det(a):
    x = torch.from_numpy(a).permute(2, 0, 1)[None].cuda()
    with torch.autocast("cuda", dtype=torch.float16): return [float(torch.sigmoid(n_(x).float()).mean()) for n_ in nets]
full = np.ones((512, 512), bool)
TARGET = {"stegastamp": "+0.040 +0.306 +0.385 +0.401 +0.452  det0 0.110 det1 0.905", "trustmark": "+0.037 +0.100 +0.088 +0.069 +0.092"}
for meth in ("stegastamp", "trustmark"):
    wm = load_watermark(meth, device="cuda"); ds = LocalisationDataset(RELEASE, "train", methods=[meth]); rng = np.random.default_rng(0)
    bases = [ds[i]["image"] for i in range(len(ds)) if not ds.meta[i]["full_watermark"] and ds[i]["mask"].mean() < 0.05][:16]
    print(f"== {meth}: REAL challenge images     {TARGET[meth]}")
    for name, fn in CAND.items():
        S, D = [], []
        for c in bases:
            msg = rng.integers(0, 2, wm.nbits); x = fn(c, lambda im: wm.embed(np.ascontiguousarray(im), msg))
            T = wm.embed(x, msg).astype(np.float32) - wm.embed(x, 1 - msg).astype(np.float32)
            S.append([s for s, _ in stat(x.astype(np.float32), T, full)])
            if meth == "stegastamp": D.append(det(x))
        print(f"   {name:36s} " + " ".join("%+.3f" % s for s in np.mean(S, 0)) + ("  det0 %.3f det1 %.3f" % tuple(np.mean(D, 0)) if D else ""), flush=True)
