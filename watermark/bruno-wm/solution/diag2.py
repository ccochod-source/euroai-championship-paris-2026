"""Synthetic-vs-real gap of a checkpoint: mIoU on our own synthetic composites (external and in-domain bases, by mask
type) versus the real train images. usage: diag2.py <method> <checkpoint> <hpf> <tags>"""
import sys, numpy as np, torch
from common import OUT
from data import Synth, make_mask
from model import Net
from wmloc.utils import miou_image
meth, ck, hpf, tags = sys.argv[1], sys.argv[2], bool(int(sys.argv[3])), tuple(sys.argv[4].split(","))
net = Net(hpf=hpf).cuda().to(memory_format=torch.channels_last).eval(); net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
@torch.no_grad()
def pred(x):
    with torch.autocast("cuda", dtype=torch.float16): return torch.sigmoid(net(x.cuda().contiguous(memory_format=torch.channels_last)).float()).cpu().numpy()
for p_in, nm in ((0.0, "external bases"), (1.0, "in-domain bases")):
    ds = Synth(meth, tags=tags, p_in=p_in); sc = {"full": [], "partial": []}
    for i in range(0, 192, 16):
        items = [ds[j] for j in range(i, i + 16)]; p = pred(torch.stack([a for a, _ in items]))
        for k, (_, y) in enumerate(items):
            sc["full" if y.mean() > 0.5 else "partial"].append(miou_image(p[k] > 0.5, y.numpy()))
    print(f"SYNTH {meth} {nm}: full-type {np.mean(sc['full']):.3f} (n={len(sc['full'])})  partial-type {np.mean(sc['partial']):.3f} (n={len(sc['partial'])})", flush=True)
