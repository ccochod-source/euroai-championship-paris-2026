"""Dump probability maps of one checkpoint on the real train and test images of its method, in the shared format
(out/{method}_{name}_{train,test}probs.npz with keys names, probs float16 N x 128 x 128).
usage: export_probs.py <method> <checkpoint> <name> <hpf 0/1>"""
import sys, numpy as np, torch
from torch.utils.data import DataLoader
from common import OUT
from data import Real
from model import Net
from wmloc.utils import miou_image
meth, ck, name, hpf = sys.argv[1], sys.argv[2], sys.argv[3], bool(int(sys.argv[4]))
net = Net(hpf=hpf).cuda().to(memory_format=torch.channels_last).eval(); net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
for split in ("train", "test"):
    names, P, G, Fu = [], [], [], []
    with torch.no_grad():
        for x, m, n, f in DataLoader(Real(split, meth), batch_size=16, num_workers=4):
            with torch.autocast("cuda", dtype=torch.float16):
                P.append(torch.sigmoid(net(x.cuda().contiguous(memory_format=torch.channels_last)).float()).cpu().numpy())
            names += list(n); G.append(m.numpy()); Fu += [bool(v) for v in f]
    P = np.concatenate(P); G = np.concatenate(G); Fu = np.array(Fu)
    np.savez_compressed(f"{OUT}/{meth}_{name}_{split}probs.npz", names=np.array(names), probs=P.astype(np.float16))
    if split == "train":
        s = np.array([miou_image(P[i] > 0.5, G[i]) for i in range(len(P))])
        print(f"EXPORT {meth} {name}: train mIoU {s.mean():.4f} (full {s[Fu].mean():.4f} / notfull {s[~Fu].mean():.4f})")
