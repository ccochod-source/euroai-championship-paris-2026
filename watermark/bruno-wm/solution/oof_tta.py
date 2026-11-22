"""Recompute out-of-fold (train) and fold-averaged (test) probability maps of a finetune.py run with shift
test-time augmentation: the image is shifted by multiples of 8 px (reflect-padded), predicted, shifted back, averaged.
usage: oof_tta.py <method> <run name> <hpf 0/1> [folds=5]   -> out/{method}_{run}tta_{train,test}probs.npz"""
import sys, numpy as np, torch, torch.nn.functional as F
from common import OUT
from finetune import load_real
from model import Net
from wmloc.utils import miou_image
meth, name, hpf = sys.argv[1], sys.argv[2], bool(int(sys.argv[3])); K = int(sys.argv[4]) if len(sys.argv) > 4 else 5
SHIFTS = [(0, 0), (8, 16), (16, 8), (24, 24), (16, 0), (0, 16), (8, 8), (24, 0)]
@torch.no_grad()
def predict(net, X, shifts):
    out = np.zeros((len(X), 128, 128), np.float32)
    for i in range(0, len(X), 8):
        x = X[i:i + 8].cuda().float(); acc = torch.zeros(len(x), 128, 128, device="cuda"); cnt = torch.zeros(128, 128, device="cuda")
        for dy, dx in shifts:
            xs = F.pad(x, (dx, 0, dy, 0), mode="reflect")[..., :512, :512] if (dy or dx) else x
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(net(xs.contiguous(memory_format=torch.channels_last)).float())
            a, b = dy // 4, dx // 4
            acc[:, :128 - a, :128 - b] += p[:, a:, b:]; cnt[:128 - a, :128 - b] += 1
        out[i:i + 8] = (acc / cnt).cpu().numpy()
    return out
X, M, names, full = load_real("train", meth); Xt, _, tnames, _ = load_real("test", meth)
rng = np.random.default_rng(0); fold = np.zeros(len(X), int)
for v in (True, False):
    idx = np.where(full == v)[0]; rng.shuffle(idx); fold[idx] = np.arange(len(idx)) % K
oof0 = np.zeros((len(X), 128, 128), np.float32); oof = oof0.copy(); test = np.zeros((len(Xt), 128, 128), np.float32)
for k in range(K):
    net = Net(hpf=hpf).cuda().to(memory_format=torch.channels_last).eval(); net.load_state_dict(torch.load(f"{OUT}/{meth}_{name}_fold{k}.pt", map_location="cpu", weights_only=True))
    va = np.where(fold == k)[0]; oof0[va] = predict(net, X[va], SHIFTS[:1]); oof[va] = predict(net, X[va], SHIFTS); test += predict(net, Xt, SHIFTS) / K
sc = lambda P: np.array([miou_image(P[i] > 0.5, M[i].numpy()) for i in range(len(X))])
s0, s1 = sc(oof0), sc(oof)
print(f"TTA {meth} {name}: out-of-fold mIoU plain {s0.mean():.4f} -> shift-TTA {s1.mean():.4f}  (full {s0[full].mean():.4f} -> {s1[full].mean():.4f}, partial {s0[~full].mean():.4f} -> {s1[~full].mean():.4f})")
np.savez_compressed(f"{OUT}/{meth}_{name}tta_trainprobs.npz", names=np.array(names), probs=oof.astype(np.float16))
np.savez_compressed(f"{OUT}/{meth}_{name}tta_testprobs.npz", names=np.array(tnames), probs=test.astype(np.float16))
