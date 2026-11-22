"""PixelSeal detector helpers: float16 crop decoding (GRN statistics kept in float32) and a convolutional mode with
LOCAL response normalisation, so that the detector can be slid over a large image.

Why: the detector is a ConvNeXt-V2; its GRN layers normalise each channel by its L2 norm over the WHOLE input. When
the detector is run on a whole image, the statistics of a small watermarked rectangle are drowned by the rest and the
message is not decoded (measured). In local mode the norm is taken over a kh x kw window of final cells (1 cell =
32 px of the detector input) centred on each cell."""
import torch
import torch.nn.functional as F

CFG = {"local": None}      # None: original global GRN. (kh, kw, Hf, Wf): local statistics, Hf x Wf = final map size


def _grn_forward(self, x):                                  # x: N x H x W x C
    xf = x.float()
    if CFG["local"] is None:
        Gx = torch.norm(xf, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
    else:
        kh, kw, Hf, Wf = CFG["local"]
        N, H, W, C = x.shape
        e = F.avg_pool2d(xf.permute(0, 3, 1, 2).square(), (H // Hf, W // Wf))          # mean square per final cell
        e = F.avg_pool2d(e, (kh, kw), stride=1, padding=(kh // 2, kw // 2), count_include_pad=False)
        G = e.sqrt()
        Nx = G / (G.mean(dim=1, keepdim=True) + 1e-6)
        Nx = F.interpolate(Nx, size=(H, W), mode="nearest").permute(0, 2, 3, 1)
    return self.gamma * (x * Nx) + self.beta + x


def patch(wm):
    grn_cls = None
    for m in wm.det.modules():
        if type(m).__name__ == "GRN":
            grn_cls = type(m); break
    grn_cls.forward = _grn_forward
    return wm


@torch.no_grad()
def crop_logits(wm, x, amp=True):
    """x: N x 3 x 256 x 256 in [0,1] -> N x 256 logits (original detector, global GRN)."""
    CFG["local"] = None
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        return wm.det(x)[:, 1:].float()


@torch.no_grad()
def ppl(wm, x, k=None, amp=True):
    """x: N x 3 x H x W (multiples of 32) -> N x 256 x H/32 x W/32 per-position logits. k: None (global GRN) or
    (kh, kw) odd local window in final cells."""
    det = wm.det; lin = det.pixel_decoder.linear
    CFG["local"] = None if k is None else (k[0], k[1], x.shape[2] // 32, x.shape[3] // 32)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        emb = det.pixel_decoder.output_upscaling(det.convnext(det.preprocess(x)))
    CFG["local"] = None
    emb = emb.float()
    return (torch.einsum("bchw,oc->bohw", emb, lin.weight.float()) + lin.bias.float()[None, :, None, None])[:, 1:]
