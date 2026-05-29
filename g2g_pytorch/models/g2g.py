"""End-to-end Groove2Groove-PT model.

Architecture:
  ContentEncoder  — 2-D CNN + BiGRU → per-timestep memory  (B, T_mem, H)
  RollDecoder     — non-autoregressive cross-attention + FiLM conditioning
                    conditioned directly on the raw 6528-d style-profile vector,
                    not on a compressed bottleneck.  Each FiLM layer in the
                    decoder has direct Linear(6528 → d) access to every
                    histogram bin, preserving gradient flow to the full profile.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from .encoders import ContentEncoder, PROFILE_FLAT_DIM
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
        memory_dim = cfg.content_rnn_hidden * (2 if cfg.content_rnn_bidirectional else 1)
        # Pass PROFILE_FLAT_DIM (6528) as style_dim so every FiLM layer and the
        # additive style_proj have direct, single-layer access to the full profile.
        self.decoder = RollDecoder(
            memory_dim=memory_dim,
            style_dim=PROFILE_FLAT_DIM,
            d_model=cfg.decoder_hidden,
            num_heads=cfg.decoder_attn_heads,
            num_layers=cfg.decoder_layers,
            time_steps=cfg.num_time_steps,
            num_pitches=cfg.num_pitches,
            dropout=cfg.decoder_dropout,
        )

    def forward(self, X: torch.Tensor, profile_flat: torch.Tensor) -> tuple:
        """
        Args:
            X:            (B, 2, 128, 128) two-channel content piano-roll.
            profile_flat: (B, 6528)        flattened style-profile histograms.
        Returns:
            (pitched_logits, drum_logits)  each (B, 128, 128).
        """
        memory = self.content_encoder(X)          # (B, T_mem, H)
        return self.decoder(memory, profile_flat) # conditions on raw 6528-d vector

    @torch.no_grad()
    def predict_rolls(self, X: torch.Tensor, profile_flat: torch.Tensor) -> tuple:
        """Return (pitched_roll, drum_roll) with values in [0, 1]."""
        p, d = self.forward(X, profile_flat)
        return torch.sigmoid(p), torch.sigmoid(d)


__all__ = ["G2GModel"]
