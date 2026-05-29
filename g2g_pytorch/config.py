"""Hyperparameters for the Groove2Groove diffusion model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # --- data -----------------------------------------------------------------
    dataset_root: str = str(Path(__file__).resolve().parents[2] / "dataset")
    train_split: str = "train"
    val_split:   str = "val"
    test_split:  str = "test"
    val_from_train_fraction: float = 0.05
    num_workers: int = 2

    # --- piano-roll shape -----------------------------------------------------
    num_pitches:    int = 128
    num_time_steps: int = 128
    in_channels:    int = 2       # (pitched-max, drum)

    # --- U-Net architecture ---------------------------------------------------
    unet_base_ch:          int   = 64
    unet_ch_mults:         tuple = (1, 2, 4, 8)     # → 64, 128, 256, 512
    unet_attn_resolutions: tuple = (32, 16)
    unet_style_dim:        int   = 256
    unet_time_dim:         int   = 1024

    # --- diffusion ------------------------------------------------------------
    diffusion_steps:  int   = 1000    # T — training noise steps
    ddim_steps:       int   = 50      # inference denoising steps
    cfg_scale:        float = 3.0     # classifier-free guidance scale
    cfg_content_p:    float = 0.1     # prob of dropping X during training
    cfg_style_p:      float = 0.1     # prob of dropping style during training

    # --- training -------------------------------------------------------------
    batch_size:   int   = 16
    lr:           float = 1e-4        # lower than BCE model; Adam default
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
    logdir:          str = "checkpoints/g2g_diffusion"
    ckpt_name:       str = "model.pt"
    submission_path: str = "submission.csv"


__all__ = ["Config"]
