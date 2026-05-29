"""Hyperparameters for the Groove2Groove profile-style model (exp6)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class Config:
    # --- data -----------------------------------------------------------------
    dataset_root: str = str(Path(__file__).resolve().parents[2] / "dataset")
    train_split:  str = "train"
    val_split:    str = "val"
    test_split:   str = "test"
    val_from_train_fraction: float = 0.05
    num_workers:  int = 2

    # --- piano-roll shape -----------------------------------------------------
    num_pitches:    int = 128
    num_time_steps: int = 128
    in_channels:    int = 2       # (pitched-max, drum)

    # --- content encoder ------------------------------------------------------
    content_cnn_channels: Tuple[int, ...] = (32, 64)
    content_cnn_kernels:  Tuple           = ((3, 3), (3, 3))
    content_cnn_pools:    Tuple           = ((2, 1), (2, 1))
    content_rnn_hidden:   int             = 200
    content_rnn_bidirectional: bool       = True

    # --- decoder --------------------------------------------------------------
    # style_dim is NOT listed here — the decoder conditions directly on the
    # raw 6528-d profile (PROFILE_FLAT_DIM), bypassing any learned bottleneck.
    decoder_hidden:     int   = 256
    decoder_attn_heads: int   = 4
    decoder_layers:     int   = 4
    decoder_dropout:    float = 0.1

    # --- loss weights ---------------------------------------------------------
    loss_lambda_cp: float = 0.5   # weight for soft chroma-CP auxiliary loss
    loss_lambda_sf: float = 0.5   # weight for soft histogram-SF auxiliary loss

    # --- training -------------------------------------------------------------
    batch_size:   int   = 16
    lr:           float = 1e-4
    weight_decay: float = 0.0
    grad_clip:    float = 1.0
    max_steps:    int   = 200_000
    log_every:    int   = 50
    val_every:    int   = 2_000
    ckpt_every:   int   = 5_000
    seed:         int   = 42

    # --- inference ------------------------------------------------------------
    velocity_threshold: float = 0.05

    # --- I/O paths ------------------------------------------------------------
    logdir:          str = "checkpoints/g2g_exp6"
    ckpt_name:       str = "model.pt"
    submission_path: str = "submission.csv"


__all__ = ["Config"]
