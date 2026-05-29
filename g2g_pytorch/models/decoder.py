"""Roll-decoder with cross-attention to the encoder memory and FiLM-style
conditioning on the style vector.

For roll-to-roll generation we don't need autoregression: the output time
grid is fixed (128 steps) and is aligned with the input grid (both come
from the same 8-bar window). So we build T=128 query vectors from a
sinusoidal positional embedding and let them attend to the content
memory; the style vector is broadcast and concatenated at every step.

Two output heads emit logits for the pitched and drum rolls.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utility.pianoroll import NUM_PITCHES, T_PER_FRAGMENT


def sinusoidal_positions(T: int, dim: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(T, dim, device=device)
    pos = torch.arange(0, T, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class FiLM(nn.Module):
    """Feature-wise linear modulation: y = gamma(s) * x + beta(s)."""

    def __init__(self, style_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.to_gb = nn.Linear(style_dim, 2 * feat_dim)
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), s: (B, S)
        gb = self.to_gb(s)                         # (B, 2D)
        g, b = gb.chunk(2, dim=-1)                 # (B, D), (B, D)
        return x * (1.0 + g.unsqueeze(1)) + b.unsqueeze(1)


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, style_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.film1 = FiLM(style_dim, d_model)
        self.film2 = FiLM(style_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, mem: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # Self-attention over output queries (no causal mask: non-AR roll output).
        h = self.norm1(q)
        h = self.film1(h, style)
        a, _ = self.self_attn(h, h, h, need_weights=False)
        q = q + self.dropout(a)
        # Cross-attention to encoder memory.
        h = self.norm2(q)
        c, _ = self.cross_attn(h, mem, mem, need_weights=False)
        q = q + self.dropout(c)
        # FFN with style FiLM.
        h = self.norm3(q)
        h = self.film2(h, style)
        q = q + self.dropout(self.ff(h))
        return q


class RollDecoder(nn.Module):
    """Non-autoregressive roll decoder.

    Returns per-cell logits of shape ``(B, NUM_PITCHES, T)`` for pitched and drum.
    """

    def __init__(
        self,
        memory_dim: int,
        style_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        time_steps: int = T_PER_FRAGMENT,
        num_pitches: int = NUM_PITCHES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.T = time_steps
        self.P = num_pitches
        self.d_model = d_model
        self.style_dim = style_dim

        self.mem_proj = nn.Linear(memory_dim, d_model)
        self.pos_proj = nn.Linear(d_model, d_model)
        self.style_proj = nn.Linear(style_dim, d_model)
        self.query_emb = nn.Parameter(torch.randn(time_steps, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            CrossAttnBlock(d_model, num_heads, style_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pitched_head = nn.Linear(d_model, num_pitches)
        self.drum_head = nn.Linear(d_model, num_pitches)

    def forward(self, memory: torch.Tensor, style: torch.Tensor) -> tuple:
        B = memory.shape[0]
        mem = self.mem_proj(memory)                    # (B, T_mem, d_model)
        pe = sinusoidal_positions(self.T, self.d_model, mem.device)  # (T, D)
        s_expand = self.style_proj(style).unsqueeze(1).expand(-1, self.T, -1)  # (B, T, d_model)
        q = self.query_emb.unsqueeze(0).expand(B, -1, -1) + self.pos_proj(pe).unsqueeze(0) + s_expand
        for blk in self.blocks:
            q = blk(q, mem, style)
        q = self.norm(q)
        # (B, T, D) -> heads -> (B, T, P) -> transpose to (B, P, T)
        pitched_logits = self.pitched_head(q).transpose(1, 2).contiguous()
        drum_logits = self.drum_head(q).transpose(1, 2).contiguous()
        return pitched_logits, drum_logits


__all__ = ["RollDecoder", "sinusoidal_positions", "FiLM", "CrossAttnBlock"]
