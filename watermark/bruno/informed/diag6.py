"""Is the embedding frame of REAL fully watermarked images exactly the 512 x 512 image?
Decoder margin when the image is decoded on a frame scaled by s about the centre (s < 1: centre crop, s > 1: reflect
padding), and for 1-D stretches (only height or only width). Control: in-domain synthetic images embedded on the frame."""
import os, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYN, KIT, RELEASE  # noqa
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
S = [0.90, 0.94, 0.97, 0.985, 1.0, 1.015, 1.03, 1.06, 1.10]

def frame(x, sy, sx):
    """x: 1x3x512x512 in [0,1]; returns the image seen through a frame scaled by (sy, sx) about the centre."""
    H = int(round(512 * sy)); W = int(round(512 * sx)); py, px = max(0, (H - 512 + 1) // 2), max(0, (W - 512 + 1) // 2)
    if py or px: x = F.pad(x, (px, px, py, py), mode="reflect")
    h, w = x.shape[-2:]; y0, x0 = (h - H) // 2, (w - W) // 2
    return x[:, :, y0:y0 + H, x0:x0 + W]

for meth in sys.argv[1:] or ["stegastamp", "trustmark", "pixelseal"]:
    wm = load_watermark(meth, device="cuda")
    ds = LocalisationDataset(RELEASE, "train", methods=[meth])
    real = [(ds[i]["image"], ds[i]["message"]) for i, m in enumerate(ds.meta) if m.get("full_watermark", False)][:40]
    Fu = np.load(f"{SYN}/{meth}_in_full.npy", mmap_mode="r"); M = np.load(f"{out}/{meth}_in_msgs.npy")
    syn = [(np.array(Fu[k, 0]), M[k, 0]) for k in range(40)]
    print(f"\n===== {meth}: decoder margin relative to the full frame, by frame scale")
    print("     scale          : " + "  ".join(f"{s:5.3f}" for s in S))
    for nm, data in (("REAL", real), ("SYN ", syn)):
        for kind in ("both", "height", "width"):
            rows = []
            for img, msg in data:
                x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).cuda(); sm = torch.from_numpy(msg.astype(np.float32) * 2 - 1).cuda()
                r = []
                for s in S:
                    lg = wm.logits(frame(x, s if kind != "width" else 1.0, s if kind != "height" else 1.0)).float().to(sm.device)
                    r.append(float((lg[0] * sm).mean()))
                rows.append(np.array(r) / max(1e-6, r[S.index(1.0)]))
            R = np.array(rows); best = np.array(S)[R.argmax(1)]
            print(f"  {nm} {kind:6s}       : " + "  ".join(f"{v:5.2f}" for v in R.mean(0)) + f"    best scale: median {np.median(best):.3f}, share at 1.0: {np.mean(best == 1.0):.2f}")
    del wm; torch.cuda.empty_cache()
