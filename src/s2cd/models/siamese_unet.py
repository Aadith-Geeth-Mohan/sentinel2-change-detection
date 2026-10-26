"""Siamese U-Net for binary bi-temporal change segmentation.

Shared encoder encodes t0 and t1; absolute-difference features at every
level feed a U-Net decoder that outputs PER-PIXEL logits (no global-pool
collapse — V1's classifier head is gone).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SiameseUNet(nn.Module):
    def __init__(self, in_channels: int = 10, base: int = 64, num_classes: int = 2):
        super().__init__()
        b = base
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.enc4 = ConvBlock(b * 4, b * 8)

        self.dec3 = ConvBlock(b * 8 + b * 4, b * 4)
        self.dec2 = ConvBlock(b * 4 + b * 2, b * 2)
        self.dec1 = ConvBlock(b * 2 + b, b)

        self.head = nn.Conv2d(b, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def _encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """t0, t1: (N, C, H, W) -> logits (N, num_classes, H, W)."""
        a1, a2, a3, a4 = self._encode(t0)
        b1, b2, b3, b4 = self._encode(t1)

        d4 = torch.abs(a4 - b4)
        d3 = torch.abs(a3 - b3)
        d2 = torch.abs(a2 - b2)
        d1 = torch.abs(a1 - b1)

        x = self.dec3(torch.cat([F.interpolate(d4, scale_factor=2, mode="bilinear",
                                               align_corners=False), d3], dim=1))
        x = self.dec2(torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear",
                                               align_corners=False), d2], dim=1))
        x = self.dec1(torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear",
                                               align_corners=False), d1], dim=1))
        return self.head(x)

    @torch.no_grad()
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())