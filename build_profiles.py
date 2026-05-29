"""One-time script: pre-compute style profiles for all training styles.

Run once before training so that every training style has a profile NPZ in
dataset/style_profiles/.  Subsequent training runs will load them instantly
via load_all_profiles() instead of recomputing at dataset-init time.

Usage:
    python build_profiles.py [--dataset-root dataset]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from g2g_pytorch.utility.data import read_manifest, load_roll_bundle
from g2g_pytorch.utility.metric import build_style_profile_from_bundles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="dataset")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    profiles_dir = root / "style_profiles"
    profiles_dir.mkdir(exist_ok=True)

    triplets = read_manifest(str(root / "manifest.csv"))

    style_bundles: dict[str, list[str]] = {}
    for t in triplets:
        if t.split != "train":
            continue
        if t.Y_path and (root / t.Y_path).exists():
            style_bundles.setdefault(t.style_tgt, []).append(str(root / t.Y_path))

    print(f"Found {len(style_bundles)} training styles with Y files.")
    built = skipped = 0
    for style, paths in style_bundles.items():
        out_path = profiles_dir / f"{style}.npz"
        if out_path.exists():
            skipped += 1
            continue
        bundles = [load_roll_bundle(p) for p in paths]
        profile = build_style_profile_from_bundles(bundles)
        np.savez_compressed(str(out_path), **profile)
        built += 1
        if built % 200 == 0:
            print(f"  {built} built, {skipped} already exist...")

    print(f"Done. {built} built, {skipped} already existed.")


if __name__ == "__main__":
    main()
