"""Hyperparameters for the PyTorch port of Groove2Groove.

A single dataclass that bundles every tunable knob. Use ``Config()`` for the
default and override individual fields with kwargs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # --- data -----------------------------------------------------------------
    dataset_root: str = str(
        Path(__file__).resolve().parents[2] / "dataset"
    )
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    # If val NPZs are missing, carve a held-out chunk from train items that are
    # actually present on disk. Fraction of present train items.
    val_from_train_fraction: float = 0.05
    num_workers: int = 2

    # --- input/output shape ---------------------------------------------------
    num_pitches: int = 128
    num_time_steps: int = 128         # 8 bars * 4 beats * 4 steps
    in_channels: int = 2              # (pitched-max, drum)
    out_pitched_channels: int = 1
    out_drum_channels: int = 1

    # --- content encoder ------------------------------------------------------
    content_cnn_channels: tuple = (32, 32)
    content_cnn_kernels: tuple = ((12, 12), (4, 4))
    content_cnn_pools: tuple = ((2, 2), (2, 4))   # (pitch_pool, time_pool)
    content_rnn_hidden: int = 200
    content_rnn_bidirectional: bool = True

    # --- style encoder --------------------------------------------------------
    style_cnn_channels: tuple = (32, 32)
    style_cnn_kernels: tuple = ((12, 12), (4, 4))
    style_cnn_pools: tuple = ((2, 2), (2, 4))
    style_1d_channels: tuple = (300, 300, 300)
    style_1d_kernels: tuple = (6, 4, 4)
    style_1d_pools: tuple = (2, 2, 2)
    style_rnn_hidden: int = 500
    style_dim: int = 256
    style_dropout: float = 0.1

    # --- decoder --------------------------------------------------------------
    decoder_hidden: int = 512
    decoder_layers: int = 2
    decoder_attn_heads: int = 8
    decoder_dropout: float = 0.1

    # --- training -------------------------------------------------------------
    batch_size: int = 16
    lr: float = 5e-4                  # lowered from 1e-3 (NaN at step ~10k otherwise)
    weight_decay: float = 1e-4
    grad_clip: float = 0.5            # tighter to survive the GRU's occasional spikes
    max_steps: int = 150_000
    log_every: int = 50
    val_every: int = 1000
    ckpt_every: int = 5000
    pos_weight: float = 10.0          # was 30 — too aggressive, caused logit blow-up
    drum_loss_weight: float = 1.0
    pitched_loss_weight: float = 1.0
    style_loss_weight: float = 0.5    # style consistency auxiliary loss weight
    seed: int = 42

    # --- inference / post-processing -----------------------------------------
    velocity_threshold: float = 0.05
    softmax_temperature: float = 1.0  # placeholder for future sampling head

    # --- I/O paths ------------------------------------------------------------
    logdir: str = "checkpoints/g2g_pytorch"
    ckpt_name: str = "model.pt"
    submission_path: str = "submission.csv"


__all__ = ["Config"]
