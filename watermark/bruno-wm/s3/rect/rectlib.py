"""Rectangle scoring by decoding the known message on a crop, and local rectangle refinement.

The watermark was embedded on a rectangle of the image: the embedder resized that rectangle to its square working
resolution. Decoding only works when the decoder sees (almost) the same rectangle, so the agreement between the decoded
bits and the known message is a sharp objective for locating the rectangle.

rect = (y0, y1, x0, x1), half-open, in pixels of the 512 x 512 image.
"""
import numpy as np
import torch
import torch.nn.functional as F


class Scorer:
    def __init__(self, wm, device="cuda", amp=True):
        self.wm, self.dev, self.amp = wm, device, amp
        self.res = wm.resolution
        self.mode = "bicubic" if wm.name == "stegastamp" else "bilinear"
        self.n_eval = 0

    def crops(self, img, rects):
        """img: 1 x 3 x H x W float [0,1] on device -> N x 3 x res x res."""
        out = [F.interpolate(img[:, :, y0:y1, x0:x1], size=(self.res, self.res), mode=self.mode, align_corners=False,
                             antialias=True) for (y0, y1, x0, x1) in rects]
        return torch.cat(out)

    @torch.no_grad()
    def logits_res(self, x):
        """x: N x 3 x res x res float [0,1] -> N x nbits logits."""
        wm = self.wm; self.n_eval += len(x)
        if wm.name == "pixelseal":
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                return wm.det(x)[:, 1:].float()
        if wm.name == "trustmark":
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                return wm.net(x * 2 - 1).float()
        xn = x.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        out = [wm._run(xn[i:i + 16], [wm.LOGITS])[0] for i in range(0, len(xn), 16)]
        return torch.from_numpy(np.concatenate(out)).float().to(x.device)

    def score(self, img, rects, sm, bs=256):
        """Returns (soft agreement in [-1,1], bit accuracy) per rect. sm: nbits tensor of +-1 on device."""
        soft, acc = [], []
        for i in range(0, len(rects), bs):
            lg = self.logits_res(self.crops(img, rects[i:i + bs]))
            soft.append((torch.tanh(lg / 2) * sm[None]).mean(1)); acc.append(((lg > 0).float() * 2 - 1 == sm[None]).float().mean(1))
        return torch.cat(soft).cpu().numpy(), torch.cat(acc).cpu().numpy()


def valid(r, H=512, W=512, min_side=8, max_aspect=None):
    y0, y1, x0, x1 = r
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W or y1 - y0 < min_side or x1 - x0 < min_side:
        return False
    if max_aspect is not None:
        h, w = y1 - y0, x1 - x0
        if max(h, w) / min(h, w) > max_aspect * 1.15:
            return False
    return True


def neighbours(r, step):
    y0, y1, x0, x1 = r; s = step
    c = [(y0 - s, y1, x0, x1), (y0 + s, y1, x0, x1), (y0, y1 - s, x0, x1), (y0, y1 + s, x0, x1),
         (y0, y1, x0 - s, x1), (y0, y1, x0 + s, x1), (y0, y1, x0, x1 - s), (y0, y1, x0, x1 + s),
         (y0 - s, y1 - s, x0, x1), (y0 + s, y1 + s, x0, x1), (y0, y1, x0 - s, x1 - s), (y0, y1, x0 + s, x1 + s),
         (y0 - s, y1 + s, x0 - s, x1 + s), (y0 + s, y1 - s, x0 + s, x1 - s),
         (y0 - s, y1 + s, x0, x1), (y0 + s, y1 - s, x0, x1), (y0, y1, x0 - s, x1 + s), (y0, y1, x0 + s, x1 - s)]
    return c


def refine(scorer, img, sm, rect, max_aspect=None, first_step=None, max_iter=200):
    """Greedy hill climbing on the soft agreement. Returns (rect, soft, acc)."""
    cur = tuple(int(v) for v in rect)
    s0, a0 = scorer.score(img, [cur], sm); cs, ca = float(s0[0]), float(a0[0])
    h, w = cur[1] - cur[0], cur[3] - cur[2]
    step = first_step or max(1, int(round(min(h, w) * 0.25)))
    it = 0
    while step >= 1 and it < max_iter:
        cands = [c for c in neighbours(cur, step) if valid(c, max_aspect=max_aspect)]
        if cands:
            s, acc = scorer.score(img, cands, sm); k = int(np.argmax(s))
            if s[k] > cs + 1e-4:
                cur, cs, ca = cands[k], float(s[k]), float(acc[k]); it += 1
                continue
        step //= 2; it += 1
    return cur, cs, ca


def rect_iou(a, b):
    iy = max(0, min(a[1], b[1]) - max(a[0], b[0])); ix = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = iy * ix; ua = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / ua if ua > 0 else 0.0
