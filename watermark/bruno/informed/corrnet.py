"""Template-informed localisation network.
Inputs: image X (B x 3 x 512 x 512, [0, 255]) and normalised template t (expected residual for the known message).
The network estimates the residual actually present in X (content suppression) and multiplies it with the template
(learned matched filter); a U-Net then aggregates the local correlation into a 128 x 128 mask.
Outputs: mask logits B x 128 x 128, residual estimate B x 3 x 512 x 512 (auxiliary loss)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RB, down, Up


class CorrNet(nn.Module):
    def __init__(self, ce=16, w=(48, 64, 128, 256, 384, 512)):
        super().__init__()
        g = torch.tensor([1., 4., 6., 4., 1.]); g = (g[:, None] * g[None]) / 256
        self.register_buffer("g5", g[None, None].repeat(3, 1, 1, 1))
        self.stem = nn.Sequential(nn.Conv2d(18, w[0], 3, padding=1, bias=False), nn.BatchNorm2d(w[0]), nn.SiLU(), RB(w[0]), RB(w[0]), RB(w[0]))
        self.toE = nn.Conv2d(w[0], ce, 1)
        self.tbr = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, ce, 3, padding=1))
        self.mix = nn.Sequential(nn.Conv2d(w[0] + 2 * ce, w[0], 1, bias=False), nn.BatchNorm2d(w[0]), nn.SiLU())
        self.d1 = down(w[0], w[1]); self.d2 = down(w[1], w[2], 3); self.d3 = down(w[2], w[3], 3)
        self.d4 = down(w[3], w[4], 3); self.d5 = down(w[4], w[5], 3)
        self.u4 = Up(w[5], w[4], w[4]); self.u3 = Up(w[4], w[3], w[3]); self.u2 = Up(w[3], w[2], w[2])
        self.head = nn.Sequential(nn.Conv2d(w[2], w[2], 3, padding=1, bias=False), nn.BatchNorm2d(w[2]), nn.SiLU(), nn.Conv2d(w[2], 1, 1))

    def forward(self, x, t):
        x = x.float()
        hp1 = (x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1)) / 4
        hp2 = (x - F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), self.g5.to(x.dtype), groups=3)) / 4
        t = t.to(x.dtype)
        inp = torch.cat([x / 127.5 - 1.0, hp1, hp2, t, hp1 * t, hp2 * t], 1)
        s0 = self.stem(inp)
        E = self.toE(s0)
        s0 = self.mix(torch.cat([s0, E * self.tbr(t), E], 1))
        s1 = self.d1(s0); s2 = self.d2(s1); s3 = self.d3(s2); s4 = self.d4(s3); s5 = self.d5(s4)
        y = self.u2(self.u3(self.u4(s5, s4), s3), s2)
        return self.head(y)[:, 0], E[:, :3]
