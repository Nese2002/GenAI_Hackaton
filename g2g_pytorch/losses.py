"""Differentiable surrogate losses aligned with the competition scoring metrics.

CP surrogate  — soft windowed chroma cosine between output and content roll.
               Mirrors content_preservation() in utility/metric.py.

SF surrogate  — soft [time_pitch, onset_drum] histogram cosine between
               output and the conditioning profile.
               Covers 6144 / 6528 (94 %) of the style_fit() profile vector;
               onset_duration and onset_velocity require note extraction and
               are omitted here.

Usage:
    from .losses import soft_chroma_cp_loss, soft_sf_loss
    cp_loss = soft_chroma_cp_loss(sigmoid(p_logits), X[:, 0])
    sf_loss = soft_sf_loss(sigmoid(p_logits), sigmoid(d_logits), profile_flat)
    loss = bce + lambda_cp * cp_loss + lambda_sf * sf_loss
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Offsets in the flattened profile vector produced by flatten_profile():
#   [time_pitch(4096) | onset_duration(256) | onset_velocity(128) | onset_drum(2048)]
_TP_START  = 0
_TP_END    = 4096
_OD_END    = 4352
_OV_END    = 4480      # onset_drum starts here
# _DR_END  = 6528

_CHROMA    = 12
_WIN_STEPS = 8         # 2-beat window: 2 beats × 4 steps/beat = 8 piano-roll columns
_TIME_BINS = 32        # = 128 columns / 4 steps-per-beat  (== metric.TIME_BINS)
_ONSET_BINS = 16       # = 128 columns / 8 bars             (== metric.ONSET_BINS)


def soft_chroma_cp_loss(
    pred_pitched: torch.Tensor,
    X_pitched: torch.Tensor,
) -> torch.Tensor:
    """1 − mean windowed chroma cosine similarity.

    Args:
        pred_pitched: (B, 128, 128) sigmoid probabilities from the pitched head.
        X_pitched:    (B, 128, 128) content piano-roll (channel 0 of the X input).
    Returns:
        Scalar loss in [0, 1].  0 = perfect chroma alignment.
    """
    B, P, T = pred_pitched.shape
    n_w = T // _WIN_STEPS

    # Fold 128 pitches → 12 chroma classes via scatter_add (vectorised, no Python loop)
    pc  = torch.arange(P, device=pred_pitched.device) % _CHROMA          # (P,)
    idx = pc.view(1, P, 1).expand(B, -1, T)                               # (B, P, T)

    pred_c = torch.zeros(B, _CHROMA, T, device=pred_pitched.device).scatter_add(1, idx, pred_pitched)
    X_c    = torch.zeros(B, _CHROMA, T, device=X_pitched.device   ).scatter_add(1, idx, X_pitched)

    # Window average → (B, C, n_w)
    pred_w = pred_c.view(B, _CHROMA, n_w, _WIN_STEPS).mean(-1)
    X_w    = X_c.view(   B, _CHROMA, n_w, _WIN_STEPS).mean(-1)

    # Cosine similarity per window — flatten (B, n_w) into batch
    pred_f = pred_w.permute(0, 2, 1).reshape(-1, _CHROMA)  # (B*n_w, 12)
    X_f    = X_w.permute(  0, 2, 1).reshape(-1, _CHROMA)
    cos    = F.cosine_similarity(pred_f, X_f, dim=1)        # (B*n_w,)
    return 1.0 - cos.mean()


def soft_sf_loss(
    pred_pitched: torch.Tensor,
    pred_drum: torch.Tensor,
    profile_flat: torch.Tensor,
) -> torch.Tensor:
    """1 − cosine similarity between soft histograms and the target profile.

    Computes soft time_pitch and onset_drum histograms from the output
    probability maps and compares against the corresponding slices of
    profile_flat.  The two histograms account for (4096 + 2048) / 6528 = 94 %
    of the full profile used by style_fit().

    Args:
        pred_pitched:  (B, 128, 128) sigmoid probs from the pitched head.
        pred_drum:     (B, 128, 128) sigmoid probs from the drum head.
        profile_flat:  (B, 6528) conditioning profile for this batch.
    Returns:
        Scalar loss in [0, 1].  0 = histogram perfectly matches the profile.
    """
    B, P, T = pred_pitched.shape   # P = T = 128

    # --- soft time_pitch histogram ---
    # Each piano-roll column c belongs to time bin c // 4 (== int(onset_beats)).
    # view(B, P, TIME_BINS, T//TIME_BINS).sum(-1) groups 4 consecutive columns.
    tp = pred_pitched.view(B, P, _TIME_BINS, T // _TIME_BINS).sum(-1)   # (B, 128, 32)
    tp = tp.permute(0, 2, 1).reshape(B, -1)                              # (B, 4096)

    # --- soft onset_drum histogram ---
    # Each column c belongs to onset bin c % 16 (16 steps per bar, 8 bars).
    # view(B, P, 8_bars, 16_steps_per_bar).sum(2) accumulates over bars.
    od = pred_drum.view(B, P, T // _ONSET_BINS, _ONSET_BINS).sum(2)     # (B, 128, 16)
    od = od.permute(0, 2, 1).reshape(B, -1)                              # (B, 2048)

    soft_hist   = torch.cat([tp, od], dim=1)                              # (B, 6144)

    target_tp   = profile_flat[:, _TP_START:_TP_END]   # (B, 4096)
    target_od   = profile_flat[:, _OV_END:]            # (B, 2048)
    target_hist = torch.cat([target_tp, target_od], dim=1)                # (B, 6144)

    # L1-normalise (mirrors style_fit normalisation)
    soft_hist   = soft_hist   / (soft_hist.sum(  dim=1, keepdim=True) + 1e-8)
    target_hist = target_hist / (target_hist.sum(dim=1, keepdim=True) + 1e-8)

    cos = F.cosine_similarity(soft_hist, target_hist, dim=1)  # (B,)
    return 1.0 - cos.mean()


__all__ = ["soft_chroma_cp_loss", "soft_sf_loss"]
