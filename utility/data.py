from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import pianoroll as pr


@dataclass
class Triplet:
    item_id: str
    style_src: str
    style_tgt: str
    split: str
    X_path: str
    Z_path: str
    Y_path: str


MANIFEST_COLUMNS = [
    "item_id",
    "style_src",
    "style_tgt",
    "split",
    "X_path",
    "Z_path",
    "Y_path",
]


def write_manifest(path: str, rows: Sequence[Triplet]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in MANIFEST_COLUMNS})


def read_manifest(path: str) -> List[Triplet]:
    out: List[Triplet] = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            out.append(Triplet(**{k: row[k] for k in MANIFEST_COLUMNS}))
    return out


def save_roll_bundle(
    path: str,
    pitched: np.ndarray,
    drum: np.ndarray,
    track_ids: Sequence[int],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pitched=pitched.astype(np.float32),
        drum=drum.astype(np.float32),
        track_ids=np.asarray(track_ids, dtype=np.int32),
    )


def load_roll_bundle(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {
            "pitched": z["pitched"].astype(np.float32),
            "drum": z["drum"].astype(np.float32),
            "track_ids": z["track_ids"].astype(np.int32),
        }


def combined_roll_from_bundle(bundle: Dict[str, np.ndarray]) -> np.ndarray:
    pitched = bundle["pitched"]
    drum = bundle["drum"]
    if pitched.size > 0:
        roll = pitched.max(axis=0)
    else:
        roll = np.zeros_like(drum)
    return np.maximum(roll, drum)


class HackathonDataset:
    def __init__(self, root: str):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.csv"
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"manifest not found at {self.manifest_path}; "
                "run data/prepare_data.ipynb first"
            )
        self.triplets: List[Triplet] = read_manifest(str(self.manifest_path))
        self._by_id = {t.item_id: t for t in self.triplets}

    def split(self, name: str) -> List[Triplet]:
        return [t for t in self.triplets if t.split == name]

    def train(self) -> List[Triplet]:
        return self.split("train")

    def val(self) -> List[Triplet]:
        return self.split("val")

    def test(self) -> List[Triplet]:
        return self.split("test")

    def test_public(self) -> List[Triplet]:
        return self.split("test_public")

    def test_hidden(self) -> List[Triplet]:
        return self.split("test_hidden")

    def get(self, item_id: str, load_Y: bool = True) -> Dict[str, object]:
        t = self._by_id[item_id]
        out: Dict[str, object] = {
            "item_id": item_id,
            "style_src": t.style_src,
            "style_tgt": t.style_tgt,
            "split": t.split,
            "X": load_roll_bundle(str(self.root / t.X_path)),
            "Z": load_roll_bundle(str(self.root / t.Z_path)),
        }
        if load_Y and t.Y_path and (self.root / t.Y_path).exists():
            out["Y"] = load_roll_bundle(str(self.root / t.Y_path))
        profile_path = self.root / "style_profiles" / f"{t.style_tgt}.npz"
        if profile_path.exists():
            with np.load(profile_path) as z:
                out["style_profile"] = {k: z[k] for k in z.files}
        return out


_TOY_STYLES = [
    "style_a",
    "style_b",
    "style_c",
    "style_d",
    "style_e",
    "style_f",
    "style_g",
    "style_h",
    "style_i",
    "style_j",
    "style_k",
    "style_l",
    "style_m",
    "style_n",
    "style_o",
    "style_p",
    "style_q",
    "style_r",
    "style_s",
    "style_t",
    "style_u",
    "style_v",
    "style_w",
    "style_x",
]

_TOY_CHORD_BANK = [
    (0, "maj"),
    (5, "maj"),
    (7, "maj"),
    (9, "min"),
    (2, "min"),
    (4, "min"),
    (11, "dim"),
]

_CHORD_INTERVALS = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
}


def _seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.uint64(seed))


def _toy_content_roll(
    item_seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, str]]]:
    rng = _seeded_rng(item_seed)
    T = pr.T_PER_FRAGMENT
    pitched = np.zeros((1, pr.NUM_PITCHES, T), dtype=np.float32)
    drum = np.zeros((pr.NUM_PITCHES, T), dtype=np.float32)
    progression = [
        _TOY_CHORD_BANK[int(rng.integers(0, len(_TOY_CHORD_BANK)))] for _ in range(4)
    ]
    chord_chart: List[Tuple[int, str]] = []
    cells_per_bar = pr.STEPS_PER_BEAT * pr.BEATS_PER_BAR
    for bar in range(pr.BARS_PER_FRAGMENT):
        root, qual = progression[bar % 4]
        chord_chart.append((root, qual))
        for iv in _CHORD_INTERVALS[qual]:
            pitch = 48 + ((root + iv) % 12)
            s = bar * cells_per_bar
            e = (bar + 1) * cells_per_bar
            pitched[0, pitch, s:e] = 0.55
        bass_pitch = 36 + (root % 12)
        for beat_off in (0, 2):
            s = bar * cells_per_bar + beat_off * pr.STEPS_PER_BEAT
            e = s + pr.STEPS_PER_BEAT
            pitched[0, bass_pitch, s:e] = 0.7
    return pitched, drum, chord_chart


def _toy_style_roll(style: str, content_chord_chart: List[Tuple[int, str]]):
    seed = abs(hash(style)) % (2**32 - 1)
    rng = _seeded_rng(seed)
    T = pr.T_PER_FRAGMENT
    cells_per_bar = pr.STEPS_PER_BEAT * pr.BEATS_PER_BAR

    onset_density = float(rng.uniform(0.2, 0.6))
    pattern = (rng.random(cells_per_bar) < onset_density).astype(np.float32)
    register_offset = int(rng.integers(0, 13))
    drum_density = float(rng.uniform(0.2, 0.7))
    drum_pattern = (rng.random(cells_per_bar) < drum_density).astype(np.float32)
    velocity_mean = float(rng.uniform(0.4, 0.95))

    def _paint(chord_chart):
        pitched = np.zeros((1, pr.NUM_PITCHES, T), dtype=np.float32)
        drum = np.zeros((pr.NUM_PITCHES, T), dtype=np.float32)
        for bar in range(pr.BARS_PER_FRAGMENT):
            root, qual = chord_chart[bar % len(chord_chart)]
            for cell in range(cells_per_bar):
                t = bar * cells_per_bar + cell
                if pattern[cell] > 0:
                    iv = _CHORD_INTERVALS[qual][cell % 3]
                    pitch = 60 + register_offset + ((root + iv) % 12)
                    pitch = int(np.clip(pitch, 0, 127))
                    pitched[0, pitch, t] = velocity_mean
                if drum_pattern[cell] > 0:
                    drum_pitch = 36 if cell % 4 == 0 else 38
                    drum[drum_pitch, t] = velocity_mean
        return pitched, drum

    demo_chart = [
        (_TOY_CHORD_BANK[(i + seed) % len(_TOY_CHORD_BANK)]) for i in range(8)
    ]
    Z_pitched, Z_drum = _paint(demo_chart)
    Y_pitched, Y_drum = _paint(content_chord_chart)
    return Z_pitched, Z_drum, Y_pitched, Y_drum


def generate_toy_dataset(
    output_dir: str,
    n_train_styles: int = 8,
    n_val_styles: int = 4,
    n_test_styles: int = 4,
    items_per_style: int = 6,
    test_public_frac: float = 0.5,
    seed: int = 0,
) -> str:
    rng = random.Random(seed)
    root = Path(output_dir)
    (root / "rolls").mkdir(parents=True, exist_ok=True)
    (root / "style_profiles").mkdir(parents=True, exist_ok=True)

    styles = list(_TOY_STYLES)
    rng.shuffle(styles)
    train_styles = styles[:n_train_styles]
    val_styles = styles[n_train_styles : n_train_styles + n_val_styles]
    test_styles = styles[
        n_train_styles + n_val_styles : n_train_styles + n_val_styles + n_test_styles
    ]
    assert not set(train_styles) & set(val_styles)
    assert not set(train_styles) & set(test_styles)
    assert not set(val_styles) & set(test_styles)

    triplets: List[Triplet] = []
    item_counter = 0

    def _emit(style_tgt: str, style_src: str, split: str):
        nonlocal item_counter
        item_id = f"{split}_{item_counter:05d}"
        item_counter += 1
        item_seed = (abs(hash(style_src)) ^ item_counter) & 0xFFFFFFFF
        X_pitched, X_drum, chord_chart = _toy_content_roll(item_seed)
        Z_pitched, Z_drum, Y_pitched, Y_drum = _toy_style_roll(style_tgt, chord_chart)

        sub = root / "rolls" / style_tgt
        sub.mkdir(parents=True, exist_ok=True)
        X_path = f"rolls/{style_tgt}/{item_id}_X.npz"
        Z_path = f"rolls/{style_tgt}/{item_id}_Z.npz"
        Y_path = f"rolls/{style_tgt}/{item_id}_Y.npz"
        save_roll_bundle(str(root / X_path), X_pitched, X_drum, [0])
        save_roll_bundle(str(root / Z_path), Z_pitched, Z_drum, [0])
        save_roll_bundle(str(root / Y_path), Y_pitched, Y_drum, [0])
        triplets.append(
            Triplet(
                item_id=item_id,
                style_src=style_src,
                style_tgt=style_tgt,
                split=split,
                X_path=X_path,
                Z_path=Z_path,
                Y_path=Y_path,
            )
        )

    for st in train_styles:
        for _ in range(items_per_style):
            src = rng.choice([s for s in train_styles if s != st] or [st])
            _emit(st, src, "train")
    for st in val_styles:
        for _ in range(items_per_style):
            src = rng.choice(train_styles)
            _emit(st, src, "val")
    for st in test_styles:
        for i in range(items_per_style):
            src = rng.choice(train_styles)
            split = (
                "test_public"
                if i < int(items_per_style * test_public_frac)
                else "test_hidden"
            )
            _emit(st, src, split)

    write_manifest(str(root / "manifest.csv"), triplets)

    from .metric import build_style_profile_from_bundles

    style_to_bundles: Dict[str, List[Dict[str, np.ndarray]]] = {}
    for t in triplets:
        if t.split == "train":
            continue
        b = load_roll_bundle(str(root / t.Y_path))
        style_to_bundles.setdefault(t.style_tgt, []).append(b)
    for st, bundles in style_to_bundles.items():
        profile = build_style_profile_from_bundles(bundles)
        np.savez_compressed(str(root / "style_profiles" / f"{st}.npz"), **profile)

    return str(root)


__all__ = [
    "Triplet",
    "MANIFEST_COLUMNS",
    "write_manifest",
    "read_manifest",
    "save_roll_bundle",
    "load_roll_bundle",
    "combined_roll_from_bundle",
    "HackathonDataset",
    "generate_toy_dataset",
]
