"""Thin wrapper around utility.metric to score model outputs on val."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .utility import metric as _metric
from .utility.pianoroll import VELOCITY_THRESHOLD


def rolls_to_bundle(
    pitched: np.ndarray,
    drum: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Wrap two 2-D rolls (P, T) into the bundle dict expected by utility.metric."""
    if pitched.ndim == 2:
        pitched = pitched[None, :, :]
    return {
        "pitched": pitched.astype(np.float32),
        "drum": drum.astype(np.float32),
        "track_ids": np.array([0], dtype=np.int32),
    }


def per_item_profile_from_Y(
    y_pitched: np.ndarray,
    y_drum: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build a histogram profile from a single Y bundle.

    Used when a precomputed style profile is unavailable for the item's
    target style (e.g. when val items are carved out of train).
    """
    return _metric.compute_histograms(rolls_to_bundle(y_pitched, y_drum))


def score_item_from_rolls(
    X_roll_combined: np.ndarray,
    pitched_out: np.ndarray,
    drum_out: np.ndarray,
    target_profile: Dict[str, np.ndarray],
) -> Dict[str, float]:
    return _metric.score_item(X_roll_combined, rolls_to_bundle(pitched_out, drum_out), target_profile)


def combined_roll_from_input(X_2ch: np.ndarray) -> np.ndarray:
    """Combine the 2-channel model input into a single roll (max of pitched and drum)."""
    return np.maximum(X_2ch[0], X_2ch[1])


def torch_logits_to_rolls(
    logits_pitched: torch.Tensor,
    logits_drum: torch.Tensor,
    threshold: float = VELOCITY_THRESHOLD,
) -> tuple:
    p = torch.sigmoid(logits_pitched).detach().cpu().numpy()
    d = torch.sigmoid(logits_drum).detach().cpu().numpy()
    p = np.where(p < threshold, 0.0, p)
    d = np.where(d < threshold, 0.0, d)
    return p.astype(np.float32), d.astype(np.float32)


__all__ = [
    "rolls_to_bundle",
    "per_item_profile_from_Y",
    "score_item_from_rolls",
    "combined_roll_from_input",
    "torch_logits_to_rolls",
]
