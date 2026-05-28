"""Thin wrapper around utility.metric to score model outputs on val."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

# Make sibling ``utility`` package importable.
# Layout: <repo_root>/g2g_pytorch/metrics.py  +  <repo_root>/utility/
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utility import metric as _metric         # type: ignore  # noqa: E402
from utility import data as _data             # type: ignore  # noqa: E402


def per_item_profile_from_Y(y_pitched: np.ndarray, y_drum: np.ndarray
                            ) -> Dict[str, np.ndarray]:
    """Build a histogram profile from a single Y bundle.

    Used when a precomputed style profile is unavailable for the item's
    target style (e.g. when val items are carved out of train, whose
    styles don't appear in ``dataset/style_profiles/``).
    """
    bundle = rolls_to_bundle(y_pitched, y_drum)
    return _metric.compute_histograms(bundle)


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


def score_item_from_rolls(
    X_roll_combined: np.ndarray,
    pitched_out: np.ndarray,
    drum_out: np.ndarray,
    target_profile: Dict[str, np.ndarray],
) -> Dict[str, float]:
    bundle = rolls_to_bundle(pitched_out, drum_out)
    return _metric.score_item(X_roll_combined, bundle, target_profile)


def combined_roll_from_input(X_2ch: np.ndarray) -> np.ndarray:
    """Reconstruct ``X``'s combined roll (max over pitched tracks, then with drum)
    from the 2-channel input we feed to the model."""
    return np.maximum(X_2ch[0], X_2ch[1])


def torch_logits_to_rolls(
    logits_pitched: torch.Tensor,
    logits_drum: torch.Tensor,
    threshold: float,
) -> tuple:
    p = torch.sigmoid(logits_pitched).detach().cpu().numpy()
    d = torch.sigmoid(logits_drum).detach().cpu().numpy()
    p = np.where(p < threshold, 0.0, p)
    d = np.where(d < threshold, 0.0, d)
    return p.astype(np.float32), d.astype(np.float32)


__all__ = [
    "rolls_to_bundle",
    "score_item_from_rolls",
    "combined_roll_from_input",
    "torch_logits_to_rolls",
]
