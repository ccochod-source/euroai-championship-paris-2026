"""Batched "expected watermark residual" on GPU: T = embed(X, m) - X (grey levels), for the three kit watermarks.
The residual of these embedders is almost entirely determined by the message (corr ~0.7-0.8 with the true residual when
re-embedding the same message on the already watermarked image, ~0 with another message), so T tells a detector exactly
which pattern to look for. The same function is used for training and test images."""
import numpy as np
import torch
import torch.nn.functional as F


class Templater:
    def __init__(self, wm):
        self.wm, self.name = wm, wm.name

    @torch.no_grad()
    def __call__(self, x, bits):
        """x: B x 3 x H x W float in [0, 255] on cuda; bits: B x nbits (0/1). Returns B x 3 x H x W float (grey levels)."""
        x = x.float() / 255; bits = bits.float()
        if self.name == "pixelseal":
            out = self.wm.model.embed(x, msgs=bits, is_video=False)
            return (out["imgs_w"].float() - x) * 255
        if self.name == "trustmark":
            tm = self.wm.tm; r = tm.model_resolution_enc
            cover = F.interpolate(x, size=(r, r), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
            res = torch.cat([tm.encoder(cover[i:i + 1], bits[i])[0] for i in range(len(x))]).clamp(-1, 1) - cover
            res = res - res.mean(dim=(2, 3), keepdim=True)
            return F.interpolate(res, size=x.shape[-2:], mode="bilinear") * 127.5
        small = F.interpolate(x, size=(400, 400), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        res = self.wm._run(small.permute(0, 2, 3, 1).cpu().numpy(), ["residual"], bits.cpu().numpy().astype(np.float32))[0]
        res = torch.from_numpy(res).to(x.device).permute(0, 3, 1, 2) * 255
        res = res - res.mean(dim=(2, 3), keepdim=True)
        return F.interpolate(res, size=x.shape[-2:], mode="bilinear", align_corners=False)


def normalise(T):
    """Per-image RMS normalisation. Returns (t, scale)."""
    s = T.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-3
    return T / s, s
