"""End-to-end Groove2Groove-PT model: content encoder + style encoder + roll decoder."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from .encoders import ContentEncoder, StyleEncoder
from .decoder import RollDecoder


class G2GModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.content_encoder = ContentEncoder(
            in_channels=cfg.in_channels,
            cnn_channels=cfg.content_cnn_channels,
            cnn_kernels=cfg.content_cnn_kernels,
            cnn_pools=cfg.content_cnn_pools,
            rnn_hidden=cfg.content_rnn_hidden,
            bidirectional=cfg.content_rnn_bidirectional,
        )
        self.style_encoder = StyleEncoder(
            in_channels=cfg.in_channels,
            cnn2d_channels=cfg.style_cnn_channels,
            cnn2d_kernels=cfg.style_cnn_kernels,
            cnn2d_pools=cfg.style_cnn_pools,
            cnn1d_channels=cfg.style_1d_channels,
            cnn1d_kernels=cfg.style_1d_kernels,
            cnn1d_pools=cfg.style_1d_pools,
            rnn_hidden=cfg.style_rnn_hidden,
            style_dim=cfg.style_dim,
            dropout=cfg.style_dropout,
        )
        # We pass a placeholder memory_dim that we'll resolve lazily by running
        # one dummy forward. Easier: pre-compute by knowing content_rnn output.
        memory_dim = cfg.content_rnn_hidden * (2 if cfg.content_rnn_bidirectional else 1)
        self.decoder = RollDecoder(
            memory_dim=memory_dim,
            style_dim=cfg.style_dim,
            d_model=cfg.decoder_hidden,
            num_heads=cfg.decoder_attn_heads,
            num_layers=cfg.decoder_layers,
            time_steps=cfg.num_time_steps,
            num_pitches=cfg.num_pitches,
            dropout=cfg.decoder_dropout,
        )

    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> tuple:
        memory = self.content_encoder(X)               # (B, T_mem, H)
        style  = self.style_encoder(Z)                 # (B, style_dim)
        return self.decoder(memory, style)             # (logits_pitched, logits_drum)

    @torch.no_grad()
    def predict_rolls(self, X: torch.Tensor, Z: torch.Tensor) -> tuple:
        """Return (pitched_roll, drum_roll) with values in [0, 1]."""
        p, d = self.forward(X, Z)
        return torch.sigmoid(p), torch.sigmoid(d)


__all__ = ["G2GModel"]
