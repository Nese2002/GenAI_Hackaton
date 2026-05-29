"""PyTorch Datasets for the hackathon piano-roll bundles.

Each sample provides:
  X            — (2, 128, 128) content piano roll
  Z            — (2, 128, 128) style example piano roll  (kept for reference)
  style_flat   — (PROFILE_FLAT_DIM,) flattened style profile histograms
  Y_pitched    — (128, 128) target pitched roll  (train/val only)
  Y_drum       — (128, 128) target drum roll      (train/val only)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .utility.data import Triplet, read_manifest, load_roll_bundle, HackathonDataset
from .utility.pianoroll import NUM_PITCHES, T_PER_FRAGMENT
from .models.unet import flatten_profile


# ---------------------------------------------------------------------------
# Bundle → 2-channel piano-roll
# ---------------------------------------------------------------------------

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
    """Collapse a bundle into a (2, num_pitches, T) float32 array.
    Channel 0: max over pitched tracks. Channel 1: drum roll.
    """
    pitched = bundle["pitched"]
    drum    = bundle["drum"]
    p = pitched.max(axis=0) if pitched.size > 0 else np.zeros((num_pitches, T), dtype=np.float32)
    p = _fit(p, num_pitches, T)
    d = _fit(drum, num_pitches, T)
    return np.stack([p, d], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Style profile loader
# ---------------------------------------------------------------------------

def load_all_profiles(root: str) -> Dict[str, torch.Tensor]:
    """Load all style profile NPZs into a dict {style_name: flat_tensor}."""
    profiles_dir = Path(root) / "style_profiles"
    out: Dict[str, torch.Tensor] = {}
    if not profiles_dir.exists():
        return out
    for f in profiles_dir.iterdir():
        if f.suffix != ".npz":
            continue
        with np.load(f) as z:
            profile = {k: z[k] for k in z.files}
        out[f.stem] = flatten_profile(profile)
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HackathonRollDataset(Dataset):
    """Yields dicts with X, Z, style_flat, and optionally Y_pitched / Y_drum."""

    def __init__(
        self,
        root: str,
        triplets: Sequence[Triplet],
        num_pitches: int = NUM_PITCHES,
        time_steps: int = T_PER_FRAGMENT,
        require_Y: bool = True,
        profiles: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        self.root        = Path(root)
        self.num_pitches = num_pitches
        self.time_steps  = time_steps
        self.require_Y   = require_Y
        self.profiles    = profiles or {}
        self.triplets: List[Triplet] = []
        skipped = 0
        for t in triplets:
            if not (self.root / t.X_path).exists():
                skipped += 1; continue
            if not (self.root / t.Z_path).exists():
                skipped += 1; continue
            if require_Y:
                if not t.Y_path or not (self.root / t.Y_path).exists():
                    skipped += 1; continue
            self.triplets.append(t)
        self._skipped = skipped

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict:
        t = self.triplets[idx]
        X = bundle_to_input(load_roll_bundle(str(self.root / t.X_path)),
                            self.num_pitches, self.time_steps)
        Z = bundle_to_input(load_roll_bundle(str(self.root / t.Z_path)),
                            self.num_pitches, self.time_steps)

        # Style profile — use pre-loaded dict, fall back to zeros
        style_flat = self.profiles.get(t.style_tgt,
                                       torch.zeros(1))  # zero = unknown style

        sample = {
            "X":          torch.from_numpy(X),
            "Z":          torch.from_numpy(Z),
            "style_flat": style_flat,
            "item_id":    t.item_id,
            "style_tgt":  t.style_tgt,
        }
        if t.Y_path and (self.root / t.Y_path).exists():
            Y = bundle_to_input(load_roll_bundle(str(self.root / t.Y_path)),
                                self.num_pitches, self.time_steps)
            sample["Y_pitched"] = torch.from_numpy(Y[0])
            sample["Y_drum"]    = torch.from_numpy(Y[1])
        return sample


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

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


def carve_val_from_train(
    root: str,
    train: Sequence[Triplet],
    fraction: float,
    seed: int,
) -> Tuple[List[Triplet], List[Triplet]]:
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


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate(batch: List[dict]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "X":          torch.stack([b["X"] for b in batch]),
        "Z":          torch.stack([b["Z"] for b in batch]),
        "style_flat": torch.stack([b["style_flat"] for b in batch]),
        "item_id":    [b["item_id"]  for b in batch],
        "style_tgt":  [b["style_tgt"] for b in batch],
    }
    if "Y_pitched" in batch[0]:
        out["Y_pitched"] = torch.stack([b["Y_pitched"] for b in batch])
        out["Y_drum"]    = torch.stack([b["Y_drum"]    for b in batch])
    return out


__all__ = [
    "Triplet", "read_manifest", "HackathonDataset",
    "bundle_to_input", "load_all_profiles", "HackathonRollDataset",
    "split_triplets", "carve_val_from_train", "collate",
]
