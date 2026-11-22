"""Where does the synthetic -> real gap come from? Evaluate a checkpoint on several kinds of inputs."""
import sys, json, numpy as np, torch
from torch.utils.data import DataLoader
from common import OUT, RELEASE
from data import Synth, Real
from model import Net
from wmloc.utils import miou_image, to_tensor
from wmloc.watermarks import load_watermark
meth, ck = sys.argv[1], sys.argv[2]; hpf = len(sys.argv) > 3 and sys.argv[3] == "1"
net = Net(hpf=hpf).cuda().to(memory_format=torch.channels_last).eval()
net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
@torch.no_grad()
def pred(x):
    with torch.autocast("cuda", dtype=torch.float16):
        return torch.sigmoid(net(x.cuda().contiguous(memory_format=torch.channels_last)).float()).cpu().numpy()
for p_in, nm in ((0.0, "synthetic external"), (1.0, "synthetic in-domain")):
    ds = Synth(meth, p_in=p_in); sc = []
    for i in range(0, 64, 16):
        xs, ys = zip(*[ds[j] for j in range(i, i + 16)]); p = pred(torch.stack(xs))
        sc += [miou_image(p[k] > 0.5, ys[k].numpy()) for k in range(16)]
    print(f"{nm}: mIoU {np.mean(sc):.4f}")
real = Real("train", meth); wm = load_watermark(meth, device="cuda"); rng = np.random.default_rng(0)
rows = {True: [], False: []}; emb = []; n = 0
for i in range(len(real)):
    x, m, name, full = real[i]
    p = pred(x[None])[0]
    rows[full].append((p.mean(), m.mean().item(), miou_image(p > 0.5, m.numpy()), p[m.numpy() > 0.5].mean() if m.sum() > 0 else -1, p[m.numpy() < 0.5].mean() if (1 - m).sum() > 0 else -1))
    if not full and n < 24:   # mostly-clean real image: watermark it ourselves, the network should fire everywhere
        img = x.permute(1, 2, 0).numpy(); w = wm.embed(np.ascontiguousarray(img), rng.integers(0, 2, wm.nbits))
        emb.append(pred(torch.from_numpy(w).permute(2, 0, 1)[None])[0].mean()); n += 1
for full in (True, False):
    r = np.array(rows[full])
    print(f"real full={full}: n={len(r)} mean prob {r[:,0].mean():.3f} (gt frac {r[:,1].mean():.3f}) mIoU {r[:,2].mean():.4f} | prob on wm pixels {r[r[:,3]>=0,3].mean():.3f} | prob on clean pixels {r[r[:,4]>=0,4].mean():.3f}")
print(f"real notfull image + OUR full embedding: mean prob {np.mean(emb):.3f}")
