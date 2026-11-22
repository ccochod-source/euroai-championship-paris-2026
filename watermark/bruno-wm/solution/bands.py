"""Band-pass helpers shared by the probes."""
import numpy as np, torch, torch.nn.functional as F
SIG = [0, 0.7, 1.5, 3, 6, 12]
def blur(x, s):
    if s <= 0: return x
    r = int(3 * s + 0.5); t = torch.arange(-r, r + 1, dtype=torch.float32, device=x.device); k = torch.exp(-t ** 2 / (2 * s * s)); k /= k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)
def bands(a):
    x = torch.from_numpy(a).permute(2, 0, 1)[None].cuda(); g = [blur(x, s) for s in SIG]
    return [(g[i] - g[i + 1])[0].permute(1, 2, 0).cpu().numpy() for i in range(len(SIG) - 1)]
def stat(x, T, region):
    out = []
    for bx, bt in zip(bands(x), bands(T)):
        a, b = bx[region], bt[region]
        out.append((float((a * b).sum() / max((b * b).sum(), 1e-9)), float(np.sqrt((b * b).mean()))))
    return out
