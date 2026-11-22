"""Do REAL images that match the kit template differ from the others in pixel statistics (sign of a generative
re-rendering, e.g. a VAE round trip)? Per image: high-frequency energy and 8/16-px grid periodicity of the high-pass
residual, for matched (detector prob > 0.8) vs unmatched (< 0.4) fully watermarked images, and for the mostly clean
not-full images of the same method."""
import os, sys, numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT, RELEASE  # noqa
from corrnet import CorrNet
from templates import Templater, normalise
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmloc.utils import downsample_mask

def stats(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F); hp = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    F = np.abs(np.fft.fft2(hp)) ** 2; F = np.fft.fftshift(F); c = 256
    tot = F.sum(); hi = F.copy(); yy, xx = np.mgrid[-256:256, -256:256]; r = np.sqrt(yy ** 2 + xx ** 2)
    peaks = lambda p: (F[c, c + 512 // p] + F[c + 512 // p, c] + F[c, c - 512 // p] + F[c - 512 // p, c]) / (4 * np.median(F[(r > 512 // p - 6) & (r < 512 // p + 6)]) + 1e-9)
    lsb = np.mean(img & 1)
    return [float(np.abs(lap).mean()), float(F[r > 180].sum() / tot), float(F[(r > 100) & (r <= 180)].sum() / tot), float(peaks(8)), float(peaks(16)), float(lsb), float(len(np.unique(img.reshape(-1, 3), axis=0)) / 262144)]

for meth in sys.argv[1:]:
    wm = load_watermark(meth, device="cuda"); templ = Templater(wm)
    net = CorrNet().cuda().eval(); net.load_state_dict(torch.load(f"out/{meth}_i1_last.pt", map_location="cpu", weights_only=True))
    ds = LocalisationDataset(RELEASE, "train", methods=[meth]); G = {"matched": [], "unmatched": [], "middle": [], "not-full (mostly clean)": []}
    for i, m in enumerate(ds.meta):
        it = ds[i]
        if not m.get("full_watermark", False):
            G["not-full (mostly clean)"].append(stats(it["image"])); continue
        g = downsample_mask(it["mask"].astype(np.float32))
        x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].cuda().float(); b = torch.from_numpy(it["message"][None].astype(np.float32)).cuda()
        with torch.no_grad():
            t, _ = normalise(templ(x, b))
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(net(x, t)[0].float())[0].cpu().numpy()
        pr = p[g > 0].mean(); G["matched" if pr > 0.8 else "unmatched" if pr < 0.4 else "middle"].append(stats(it["image"]))
    print(f"\n{meth}:  mean |laplacian| , share of HF energy (very high / high band) , 8-px grid peak , 16-px grid peak , LSB mean , distinct colours / pixels")
    for k, v in G.items():
        if v:
            v = np.array(v); print(f"  {k:24s} n={len(v):3d} : " + "  ".join(f"{a:8.4f}" for a in np.median(v, 0)))
