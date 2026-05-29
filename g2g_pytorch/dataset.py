"""PyTorch Datasets for the hackathon piano-roll bundles.

The on-disk format is documented in ``utility/data.py``. Every item id has
three NPZ bundles: ``X`` (content), ``Z`` (style example) and ``Y`` (the
ground-truth re-styled accompaniment, present for ``train`` and ``val``).

For the model we squash each bundle to a 2-channel image:

    channel 0 — max over pitched tracks
    channel 1 — drum roll

This loses per-track identity but the scoring metric does too (it sums
over tracks, and only ``is_drum`` matters in the histograms).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Make sibling ``utility`` importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hackaton.g2g_pytorch.utility.data import (
    Triplet,
    read_manifest,
    load_roll_bundle,
    HackathonDataset,
)
from hackaton.g2g_pytorch.utility.pianoroll import NUM_PITCHES, T_PER_FRAGMENT


# --------------------------------------------------------------------------
# bundle helpers
# --------------------------------------------------------------------------


def _fit(roll: np.ndarray, num_pitches: int, T: int) -> np.ndarray:
    if roll.shape != (num_pitches, T):
        out = np.zeros((num_pitches, T), dtype=np.float32)
        h = min(num_pitches, roll.shape[0])
        w = min(T, roll.shape[1])
        out[:h, :w] = roll[:h, :w]
        return out
    return roll


def bundle_to_input(
    bundle: Dict[str, np.ndarray],
    num_pitches: int = NUM_PITCHES,
    T: int = T_PER_FRAGMENT,
) -> np.ndarray:
    """Collapse a bundle into a ``(2, num_pitches, T)`` float32 array."""
    pitched = bundle["pitched"]
    drum = bundle["drum"]
    p = pitched.max(axis=0) if pitched.size > 0 else np.zeros((num_pitches, T), dtype=np.float32)
    p = _fit(p, num_pitches, T)
    d = _fit(drum, num_pitches, T)
    return np.stack([p, d], axis=0).astype(np.float32)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


class HackathonRollDataset(Dataset):
    """Yields ``(X_input, Z_input, Y_pitched, Y_drum, item_id)``.

    ``Y_*`` is absent for test items (no ground truth). Items whose NPZ
    files are missing on disk are filtered out at construction time.
    """

    def __init__(
        self,
        root: str,
        triplets: Sequence[Triplet],
        num_pitches: int = NUM_PITCHES,
        time_steps: int = T_PER_FRAGMENT,
        require_Y: bool = True,
    ) -> None:
        self.root = Path(root)
        self.num_pitches = num_pitches
        self.time_steps = time_steps
        self.require_Y = require_Y
        self.triplets: List[Triplet] = []
        skipped = 0
        for t in triplets:
            if not (self.root / t.X_path).exists():
                skipped += 1
                continue
            if not (self.root / t.Z_path).exists():
                skipped += 1
                continue
            if require_Y:
                if not t.Y_path or not (self.root / t.Y_path).exists():
                    skipped += 1
                    continue
            self.triplets.append(t)
        self._skipped = skipped

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        t = self.triplets[idx]
        X = bundle_to_input(load_roll_bundle(str(self.root / t.X_path)),
                            self.num_pitches, self.time_steps)
        Z = bundle_to_input(load_roll_bundle(str(self.root / t.Z_path)),
                            self.num_pitches, self.time_steps)
        sample = {
            "X": torch.from_numpy(X),
            "Z": torch.from_numpy(Z),
            "item_id": t.item_id,
            "style_tgt": t.style_tgt,
        }
        if t.Y_path and (self.root / t.Y_path).exists():
            Y = bundle_to_input(load_roll_bundle(str(self.root / t.Y_path)),
                                self.num_pitches, self.time_steps)
            sample["Y_pitched"] = torch.from_numpy(Y[0])
            sample["Y_drum"] = torch.from_numpy(Y[1])
        return sample


# --------------------------------------------------------------------------
# split helpers
# --------------------------------------------------------------------------


def split_triplets(
    triplets: Sequence[Triplet],
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
) -> Dict[str, List[Triplet]]:
    return {
        "train": [t for t in triplets if t.split == train_split],
        "val":   [t for t in triplets if t.split == val_split],
        "test":  [t for t in triplets if t.split == test_split],
    }

# not needed
def carve_val_from_train(
    root: str,
    train: Sequence[Triplet],
    fraction: float,
    seed: int,
) -> Tuple[List[Triplet], List[Triplet]]:
    """If no val items are available locally, hold out a slice of train.

    Returns ``(remaining_train, held_out_val)``. Only items whose X, Y, Z
    files are present on disk are considered.
    """
    rng = random.Random(seed)
    present: List[Triplet] = [
        t for t in train
        if (Path(root) / t.X_path).exists()
        and (Path(root) / t.Z_path).exists()
        and t.Y_path
        and (Path(root) / t.Y_path).exists()
    ]
    rng.shuffle(present)
    k = max(1, int(round(len(present) * fraction)))
    return present[k:], present[:k]


def collate(batch: List[dict]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {
        "X": torch.stack([b["X"] for b in batch], dim=0),
        "Z": torch.stack([b["Z"] for b in batch], dim=0),
        "item_id": [b["item_id"] for b in batch],
        "style_tgt": [b["style_tgt"] for b in batch],
    }
    if "Y_pitched" in batch[0]:
        out["Y_pitched"] = torch.stack([b["Y_pitched"] for b in batch], dim=0)
        out["Y_drum"] = torch.stack([b["Y_drum"] for b in batch], dim=0)
    return out


__all__ = [
    # re-exported from utility.data
    "Triplet",
    "read_manifest",
    "HackathonDataset",
    # local
    "bundle_to_input",
    "HackathonRollDataset",
    "split_triplets",
    "carve_val_from_train",
    "collate",
]
