"""Minimal AutoencoderKL (same parameter names as the 'diffusers' checkpoints) so that public VAE weights can be loaded
with plain PyTorch + safetensors. Used to simulate an image passing through a generative model's autoencoder."""
import json, torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import load_file


class Res(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, ci, eps=1e-6); self.conv1 = nn.Conv2d(ci, co, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, co, eps=1e-6); self.conv2 = nn.Conv2d(co, co, 3, padding=1)
        if ci != co: self.conv_shortcut = nn.Conv2d(ci, co, 1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x))); h = self.conv2(F.silu(self.norm2(h)))
        return (self.conv_shortcut(x) if hasattr(self, "conv_shortcut") else x) + h


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, c, eps=1e-6)
        self.to_q = nn.Linear(c, c); self.to_k = nn.Linear(c, c); self.to_v = nn.Linear(c, c); self.to_out = nn.ModuleList([nn.Linear(c, c)])

    def forward(self, x):
        b, c, h, w = x.shape; y = self.group_norm(x).view(b, c, h * w).transpose(1, 2)
        y = F.scaled_dot_product_attention(self.to_q(y)[:, None], self.to_k(y)[:, None], self.to_v(y)[:, None])[:, 0]
        return x + self.to_out[0](y).transpose(1, 2).reshape(b, c, h, w)


class Mid(nn.Module):
    def __init__(self, c):
        super().__init__(); self.resnets = nn.ModuleList([Res(c, c), Res(c, c)]); self.attentions = nn.ModuleList([Attn(c)])

    def forward(self, x):
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class ConvWrap(nn.Module):
    def __init__(self, c, stride, pad):
        super().__init__(); self.conv = nn.Conv2d(c, c, 3, stride=stride, padding=pad)


class Down(nn.Module):
    def __init__(self, ci, co, n, down):
        super().__init__(); self.resnets = nn.ModuleList([Res(ci if i == 0 else co, co) for i in range(n)])
        if down: self.downsamplers = nn.ModuleList([ConvWrap(co, 2, 0)])

    def forward(self, x):
        for r in self.resnets: x = r(x)
        if hasattr(self, "downsamplers"): x = self.downsamplers[0].conv(F.pad(x, (0, 1, 0, 1)))
        return x


class Up(nn.Module):
    def __init__(self, ci, co, n, up):
        super().__init__(); self.resnets = nn.ModuleList([Res(ci if i == 0 else co, co) for i in range(n)])
        if up: self.upsamplers = nn.ModuleList([ConvWrap(co, 1, 1)])

    def forward(self, x):
        for r in self.resnets: x = r(x)
        if hasattr(self, "upsamplers"): x = self.upsamplers[0].conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))
        return x


class Encoder(nn.Module):
    def __init__(self, ch, n, lat):
        super().__init__(); self.conv_in = nn.Conv2d(3, ch[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([Down(ch[max(i - 1, 0)], ch[i], n, i < len(ch) - 1) for i in range(len(ch))])
        self.mid_block = Mid(ch[-1]); self.conv_norm_out = nn.GroupNorm(32, ch[-1], eps=1e-6); self.conv_out = nn.Conv2d(ch[-1], 2 * lat, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for b in self.down_blocks: x = b(x)
        return self.conv_out(F.silu(self.conv_norm_out(self.mid_block(x))))


class Decoder(nn.Module):
    def __init__(self, ch, n, lat):
        super().__init__(); r = ch[::-1]; self.conv_in = nn.Conv2d(lat, r[0], 3, padding=1); self.mid_block = Mid(r[0])
        self.up_blocks = nn.ModuleList([Up(r[max(i - 1, 0)], r[i], n + 1, i < len(r) - 1) for i in range(len(r))])
        self.conv_norm_out = nn.GroupNorm(32, r[-1], eps=1e-6); self.conv_out = nn.Conv2d(r[-1], 3, 3, padding=1)

    def forward(self, z):
        x = self.mid_block(self.conv_in(z))
        for b in self.up_blocks: x = b(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__(); ch = cfg["block_out_channels"]; n = cfg["layers_per_block"]; lat = cfg["latent_channels"]; self.lat = lat
        self.encoder = Encoder(ch, n, lat); self.decoder = Decoder(ch, n, lat)
        if cfg.get("use_quant_conv", True): self.quant_conv = nn.Conv2d(2 * lat, 2 * lat, 1)
        if cfg.get("use_post_quant_conv", True): self.post_quant_conv = nn.Conv2d(lat, lat, 1)

    @torch.no_grad()
    def roundtrip(self, x):
        """x: B x 3 x H x W float in [0, 255] -> same range, encoded to the posterior mean and decoded."""
        h = self.encoder(x / 127.5 - 1.0)
        if hasattr(self, "quant_conv"): h = self.quant_conv(h)
        z = h[:, :self.lat]
        if hasattr(self, "post_quant_conv"): z = self.post_quant_conv(z)
        return ((self.decoder(z) + 1.0) * 127.5).clamp(0, 255)


def load_vae(folder):
    cfg = json.load(open(f"{folder}/config.json")); sd = load_file(f"{folder}/diffusion_pytorch_model.safetensors"); out = {}
    ren = {"query": "to_q", "key": "to_k", "value": "to_v", "proj_attn": "to_out.0"}
    for k, v in sd.items():
        p = k.split(".")
        if "attentions" in p:
            if p[-2] in ren: p[-2:-1] = ren[p[-2]].split(".")
            if p[-1] == "weight" and v.ndim == 4 and p[-2] != "group_norm": v = v[:, :, 0, 0]
        out[".".join(p)] = v
    m = VAE(cfg); m.load_state_dict(out, strict=True)
    return m.eval()
