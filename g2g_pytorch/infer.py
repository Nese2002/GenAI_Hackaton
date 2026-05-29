"""Run a trained model on the test split and write a Kaggle-format CSV.

Usage:
    cd code
    python -m g2g_pytorch.infer \
        --dataset-root ../dataset \
        --ckpt runs/exp1/model.pt \
        --output submission.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .utility import submission as _sub
from .utility import pianoroll as _pr
from .utility.pianoroll import VELOCITY_THRESHOLD

from .config import Config
from .dataset import HackathonRollDataset, collate, read_manifest, split_triplets
from .models import G2GModel


_LOGGER = logging.getLogger("g2g_pytorch.infer")


def load_model(ckpt_path: str, device: torch.device) -> tuple:
    ck = torch.load(ckpt_path, map_location=device)
    cfg_dict = ck.get("cfg", {})
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config().__dict__})
    model = G2GModel(cfg).to(device)
    with torch.no_grad():
        dummy = torch.zeros(1, cfg.in_channels, cfg.num_pitches, cfg.num_time_steps, device=device)
        model(dummy, dummy)  # lazy-builds GRU and VQ codebook
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


def roll_to_notes_rows(item_id: str, pitched: np.ndarray, drum: np.ndarray,
                       threshold: float) -> List[dict]:
    """Convert one item's (P, T) predicted rolls into per-note dicts."""
    notes: List[_pr.Note] = []
    notes.extend(_pr.roll_to_notes(pitched, threshold=threshold, is_drum=False, track_id=0))
    notes.extend(_pr.roll_to_notes(drum, threshold=threshold, is_drum=True, track_id=999))
    if not notes:
        # Submission requires at least one row per item; emit a quiet placeholder
        # so the row exists. The metric will score it poorly but the CSV is accepted.
        notes.append(
            _pr.Note(pitch=60, onset_beats=0.0, duration_beats=0.25, velocity=1,
                     is_drum=False, track_id=0)
        )
    return list(_sub.notes_to_rows(item_id, notes))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--output", type=str, default="submission.csv")
    ap.add_argument("--split", type=str, default="test",
                    help="Which manifest split to predict (default: test)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override velocity threshold for note extraction.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOGGER.info(f"Using device: {device}")

    model, cfg = load_model(args.ckpt, device)
    cfg.dataset_root = args.dataset_root
    threshold = args.threshold if args.threshold is not None else cfg.velocity_threshold

    triplets = read_manifest(str(Path(cfg.dataset_root) / "manifest.csv"))
    split_map = split_triplets(triplets,
                               train_split=cfg.train_split,
                               val_split=cfg.val_split,
                               test_split=cfg.test_split)
    use = split_map.get(args.split, [])
    if not use:
        raise RuntimeError(f"No items in split '{args.split}'")

    ds = HackathonRollDataset(
        cfg.dataset_root, use, cfg.num_pitches, cfg.num_time_steps, require_Y=False
    )
    if len(ds) == 0:
        raise RuntimeError(
            f"No NPZ files on disk for split '{args.split}'. "
            f"Re-download the dataset and try again."
        )
    _LOGGER.info(f"Predicting on {len(ds)} items ({ds._skipped} skipped).")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )

    all_rows: List[dict] = []
    written_ids = set()
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            Z = batch["Z"].to(device)
            lp, ld, _ = model(X, Z)
            p = torch.sigmoid(lp).cpu().numpy().astype(np.float32)
            d = torch.sigmoid(ld).cpu().numpy().astype(np.float32)
            p[p < threshold] = 0.0
            d[d < threshold] = 0.0
            for i, iid in enumerate(batch["item_id"]):
                rows = roll_to_notes_rows(iid, p[i], d[i], threshold=threshold)
                all_rows.extend(rows)
                written_ids.add(iid)

    # Every requested item must appear in the CSV, even if the model produced nothing.
    expected_ids = [t.item_id for t in use]
    for iid in expected_ids:
        if iid not in written_ids:
            all_rows.append({
                "item_id": iid, "track_id": 0, "is_drum": 0, "pitch": 60,
                "onset_beats": 0.0, "duration_beats": 0.25, "velocity": 1,
            })

    out_path = _sub.write_submission(args.output, all_rows)
    _LOGGER.info(f"Wrote submission to {out_path}")
    df = _sub.validate_submission(out_path, required_item_ids=expected_ids)
    _LOGGER.info(f"Validated submission: {len(df)} note rows across "
                 f"{df['item_id'].nunique() if len(df) else 0} items.")


if __name__ == "__main__":
    main()
