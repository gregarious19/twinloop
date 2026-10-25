"""Networks for v2 (SPEC_V2 §8): GapfillNet (3D U-Net), LumenNet (2D U-Net + transformer)."""

from __future__ import annotations

import torch
from torch import nn


def _block(cin: int, cout: int, dim: int) -> nn.Sequential:
    conv = nn.Conv3d if dim == 3 else nn.Conv2d
    norm = nn.InstanceNorm3d if dim == 3 else nn.InstanceNorm2d
    return nn.Sequential(
        conv(cin, cout, 3, padding=1),
        norm(cout, affine=True),
        nn.LeakyReLU(0.01),
        conv(cout, cout, 3, padding=1),
        norm(cout, affine=True),
        nn.LeakyReLU(0.01),
    )


class _UNet(nn.Module):
    """Encoder-decoder with `len(channels) - 1` stride-2 downsamples and skip concatenation."""

    def __init__(self, cin: int, cout: int, channels: tuple[int, ...], dim: int):
        super().__init__()
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        up = nn.ConvTranspose3d if dim == 3 else nn.ConvTranspose2d
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        prev = cin
        for i, c in enumerate(channels):
            self.enc.append(_block(prev, c, dim))
            if i < len(channels) - 1:
                self.down.append(conv(c, c, 2, stride=2))
            prev = c
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c in reversed(channels[:-1]):
            self.up.append(up(prev, c, 2, stride=2))
            self.dec.append(_block(2 * c, c, dim))
            prev = c
        self.head = conv(channels[0], cout, 1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.down):
                skips.append(x)
                x = self.down[i](x)
        return x, skips

    def decode(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for up, blk, sk in zip(self.up, self.dec, reversed(skips), strict=True):
            x = blk(torch.cat([up(x), sk], dim=1))
        return self.head(x)


class GapfillNet(nn.Module):
    """(B, 3, 64, 64, 64) [image, mask, seed heatmap] -> (B, 1, 64, 64, 64) vessel-mask logits."""

    def __init__(self, channels: tuple[int, ...] = (16, 32, 64, 128)):
        super().__init__()
        self.net = _UNet(3, 1, tuple(channels), 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, skips = self.net.encode(x)
        return self.net.decode(z, skips)


class LumenNet(nn.Module):
    """(B, T, 1, 48, 48) straightened slices -> (B, T, 1, 48, 48) lumen logits, T <= 128."""

    MAX_T = 128

    def __init__(self, channels: tuple[int, ...] = (16, 32, 64, 128)):
        super().__init__()
        channels = tuple(channels)
        d = channels[-1]
        self.net = _UNet(1, 1, channels, 2)
        self.pos = nn.Parameter(torch.zeros(self.MAX_T, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=4, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t = x.shape[:2]
        if t > self.MAX_T:
            raise ValueError(f"LumenNet takes at most {self.MAX_T} slices, got {t}")
        z, skips = self.net.encode(x.reshape(b * t, *x.shape[2:]))
        tok = z.mean(dim=(2, 3)).reshape(b, t, -1) + self.pos[:t]
        tok = self.transformer(tok, src_key_padding_mask=pad_mask)
        z = z + tok.reshape(b * t, -1, 1, 1)
        out = self.net.decode(z, skips)
        return out.reshape(b, t, *out.shape[1:])
