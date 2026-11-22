"""Residual U-Net: 512x512 RGB in, 128x128 logits out. Full-resolution stem (no early pooling) to keep the
low-amplitude watermark residual, deep encoder for context."""
import torch, torch.nn as nn, torch.nn.functional as F


class RB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.silu(self.b1(self.c1(x)))
        return F.silu(x + self.b2(self.c2(y)))


def down(ci, co, n=2):
    return nn.Sequential(nn.Conv2d(ci, co, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(co), nn.SiLU(), *[RB(co) for _ in range(n)])


class Up(nn.Module):
    def __init__(self, ci, cs, co, n=2):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(ci + cs, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.SiLU(), *[RB(co) for _ in range(n)])

    def forward(self, x, s):
        return self.f(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))


class Net(nn.Module):
    def __init__(self, cin=3, w=(32, 64, 128, 256, 384, 512)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, w[0], 3, padding=1, bias=False), nn.BatchNorm2d(w[0]), nn.SiLU(), RB(w[0]))   # 512
        self.d1 = down(w[0], w[1])      # 256
        self.d2 = down(w[1], w[2], 3)   # 128
        self.d3 = down(w[2], w[3], 3)   # 64
        self.d4 = down(w[3], w[4], 3)   # 32
        self.d5 = down(w[4], w[5], 3)   # 16
        self.u4 = Up(w[5], w[4], w[4])
        self.u3 = Up(w[4], w[3], w[3])
        self.u2 = Up(w[3], w[2], w[2])
        self.head = nn.Sequential(nn.Conv2d(w[2], w[2], 3, padding=1, bias=False), nn.BatchNorm2d(w[2]), nn.SiLU(), nn.Conv2d(w[2], 1, 1))

    def forward(self, x):                # x: uint8 or float B x 3 x 512 x 512 in [0, 255]
        x = x.float() / 127.5 - 1.0
        s0 = self.stem(x); s1 = self.d1(s0); s2 = self.d2(s1); s3 = self.d3(s2); s4 = self.d4(s3); s5 = self.d5(s4)
        y = self.u2(self.u3(self.u4(s5, s4), s3), s2)
        return self.head(y)[:, 0]
