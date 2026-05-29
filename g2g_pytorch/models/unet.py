"""Conditional Denoising U-Net for piano-roll style transfer.

Architecture
------------
Input  : (B, 4, 128, 128)  — 2ch noisy Y  ||  2ch content X (concatenated)
Output : (B, 2, 128, 128)  — predicted noise for pitched + drum channels

Conditioning
  • Time step t   : sinusoidal embedding → MLP → added to every ResBlock
  • Style profile : 4 flattened histograms → MLP → FiLM in every ResBlock

Channel sizes at each resolution:
  128×128 → 64ch
   64×64  → 128ch
   32×32  → 256ch  (+ self-attention)
   16×16  → 512ch  (+ self-attention)
  bottleneck 16×16 → 512ch
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utility.metric import (
    TIME_BINS, PITCH_BINS, DURATION_BINS,
    VELOCITY_BINS, ONSET_BINS, DRUM_BINS,
)

# ---------------------------------------------------------------------------
# Style profile: flat input size
# ---------------------------------------------------------------------------

PROFILE_FLAT_DIM = (
    TIME_BINS * PITCH_BINS          # time_pitch:      32×128 = 4096
    + ONSET_BINS * DURATION_BINS    # onset_duration:  16×16  =  256
    + ONSET_BINS * VELOCITY_BINS    # onset_velocity:  16×8   =  128
    + ONSET_BINS * DRUM_BINS        # onset_drum:      16×128 = 2048
)                                   # total: 6528


def flatten_profile(profile: dict) -> torch.Tensor:
    """Concatenate the 4 histogram arrays into a single 1-D float32 tensor."""
    parts = [
        torch.as_tensor(profile["time_pitch"],     dtype=torch.float32).flatten(),
        torch.as_tensor(profile["onset_duration"], dtype=torch.float32).flatten(),
        torch.as_tensor(profile["onset_velocity"], dtype=torch.float32).flatten(),
        torch.as_tensor(profile["onset_drum"],     dtype=torch.float32).flatten(),
    ]
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _groups(channels: int, max_g: int = 32) -> int:
    """Largest divisor of channels that is ≤ max_g."""
    for g in range(max_g, 0, -1):
        if channels % g == 0:
            return g
    return 1


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for integer timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FiLM2D(nn.Module):
    """FiLM: y = (1 + γ(s)) * x + β(s)  applied over (B, C, H, W)."""

    def __init__(self, style_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(style_dim, 2 * channels)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        g, b = self.proj(s).chunk(2, dim=-1)
        return x * (1.0 + g[:, :, None, None]) + b[:, :, None, None]


class ResBlock(nn.Module):
    """Conv ResBlock with time-step addition and style FiLM."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_dim: int,
        style_dim: int,
    ) -> None:
        super().__init__()
        g1 = _groups(in_ch)
        g2 = _groups(out_ch)
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.film  = FiLM2D(style_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.res   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act   = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.film(h, style)
        h = self.conv2(h)
        return h + self.res(x)


class SelfAttention2D(nn.Module):
    """Multi-head self-attention over spatial dimensions (flattened H×W)."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm  = nn.GroupNorm(_groups(channels), channels)
        self.attn  = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)   # (B, HW, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.permute(0, 2, 1).view(B, C, H, W)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """Conditional denoising U-Net.

    Args:
        in_channels : noisy Y channels (2)
        cond_channels: X content channels (2) — concatenated with noisy Y
        base_ch     : channel multiplier for each resolution level
        style_dim   : output dimension of the style profile MLP
        time_dim    : dimension of the time embedding MLP output
    """

    def __init__(
        self,
        in_channels: int = 2,
        cond_channels: int = 2,
        ch_mults: tuple = (1, 2, 4, 8),
        base_ch: int = 64,
        style_dim: int = 256,
        time_dim: int = 1024,
        attn_resolutions: tuple = (32, 16),
    ) -> None:
        super().__init__()
        self.style_dim = style_dim

        # ---- time embedding ----
        sin_dim = 256
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(sin_dim),
            nn.Linear(sin_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ---- style profile encoder ----
        self.style_enc = nn.Sequential(
            nn.Linear(PROFILE_FLAT_DIM, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, style_dim),
        )

        # ---- U-Net ----
        total_in = in_channels + cond_channels   # 4
        channels = [base_ch * m for m in ch_mults]  # e.g. [64, 128, 256, 512]

        # stem
        self.stem = nn.Conv2d(total_in, channels[0], 3, padding=1)

        # encoder
        self.enc_blocks: nn.ModuleList = nn.ModuleList()
        self.downsamples: nn.ModuleList = nn.ModuleList()
        ch_in = channels[0]
        self.skip_chs: List[int] = []
        for i, ch in enumerate(channels):
            blks = nn.ModuleList([
                ResBlock(ch_in, ch, time_dim, style_dim),
                ResBlock(ch, ch, time_dim, style_dim),
            ])
            # spatial size at this level
            res = 128 // (2 ** i)
            if res in attn_resolutions:
                blks.append(SelfAttention2D(ch))
            self.enc_blocks.append(blks)
            self.skip_chs.append(ch)
            ch_in = ch
            if i < len(channels) - 1:
                self.downsamples.append(nn.Conv2d(ch, ch, 4, stride=2, padding=1))
            else:
                self.downsamples.append(nn.Identity())

        # bottleneck
        self.mid = nn.ModuleList([
            ResBlock(channels[-1], channels[-1], time_dim, style_dim),
            SelfAttention2D(channels[-1]),
            ResBlock(channels[-1], channels[-1], time_dim, style_dim),
        ])

        # decoder
        self.dec_blocks: nn.ModuleList = nn.ModuleList()
        self.upsamples: nn.ModuleList = nn.ModuleList()
        for i, ch in enumerate(reversed(channels)):
            skip_ch = self.skip_chs[-(i + 1)]
            blks = nn.ModuleList([
                ResBlock(ch + skip_ch, ch, time_dim, style_dim),
                ResBlock(ch, ch, time_dim, style_dim),
            ])
            res = 16 * (2 ** i)
            if res in attn_resolutions:
                blks.append(SelfAttention2D(ch))
            self.dec_blocks.append(blks)
            if i < len(channels) - 1:
                out_ch = channels[-(i + 2)]
                self.upsamples.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(ch, out_ch, 3, padding=1),
                    )
                )
            else:
                self.upsamples.append(nn.Identity())

        # output head
        g = _groups(channels[0])
        self.out = nn.Sequential(
            nn.GroupNorm(g, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, 1),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x_noisy:   torch.Tensor,   # (B, 2, H, W)
        x_content: torch.Tensor,   # (B, 2, H, W)  — X or zeros for uncond
        style_flat: torch.Tensor,  # (B, PROFILE_FLAT_DIM) or zeros for uncond
        t:         torch.Tensor,   # (B,) integer timesteps
    ) -> torch.Tensor:
        t_emb   = self.time_emb(t)              # (B, time_dim)
        style   = self.style_enc(style_flat)    # (B, style_dim)

        x = torch.cat([x_noisy, x_content], dim=1)  # (B, 4, H, W)
        x = self.stem(x)

        # Encoder
        skips = []
        for i, (blks, ds) in enumerate(zip(self.enc_blocks, self.downsamples)):
            for blk in blks:
                if isinstance(blk, SelfAttention2D):
                    x = blk(x)
                else:
                    x = blk(x, t_emb, style)
            skips.append(x)
            x = ds(x)

        # Bottleneck
        for blk in self.mid:
            if isinstance(blk, SelfAttention2D):
                x = blk(x)
            else:
                x = blk(x, t_emb, style)

        # Decoder
        for i, (blks, us) in enumerate(zip(self.dec_blocks, self.upsamples)):
            skip = skips[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            for blk in blks:
                if isinstance(blk, SelfAttention2D):
                    x = blk(x)
                else:
                    x = blk(x, t_emb, style)
            x = us(x)

        return self.out(x)


__all__ = ["UNet", "flatten_profile", "PROFILE_FLAT_DIM"]
