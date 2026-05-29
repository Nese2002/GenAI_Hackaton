"""Train the G2G profile-style-encoder model for piano-roll style transfer.

Usage:
    python -m g2g_pytorch.train --dataset-root ../dataset --logdir runs/exp6
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .config import Config
from .dataset import (
    HackathonRollDataset, carve_val_from_train, collate,
    load_all_profiles, read_manifest, split_triplets,
)
from .models.g2g import G2GModel
from .metrics import (
    combined_roll_from_input, score_item_from_rolls, per_item_profile_from_Y,
)

_LOGGER = logging.getLogger("g2g_pytorch.train")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_style_profiles_np(root: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load style profiles as numpy dicts for metric scoring."""
    profiles_dir = Path(root) / "style_profiles"
    out: Dict[str, Dict[str, np.ndarray]] = {}
    if not profiles_dir.exists():
        return out
    for f in profiles_dir.iterdir():
        if f.suffix != ".npz":
            continue
        with np.load(f) as z:
            out[f.stem] = {k: z[k] for k in z.files}
    return out


def evaluate(
    model: G2GModel,
    loader: DataLoader,
    device: torch.device,
    profiles_np: Dict[str, Dict[str, np.ndarray]],
    max_items: int = 50,
) -> Dict[str, float]:
    model.eval()
    scores: List[Dict[str, float]] = []
    n = 0

    with torch.no_grad():
        for batch in loader:
            X           = batch["X"].to(device)
            profile_flat = batch["style_flat"].to(device)

            pitched, drum = model.predict_rolls(X, profile_flat)
            pitched_np = pitched.cpu().numpy()
            drum_np    = drum.cpu().numpy()
            X_np  = batch["X"].numpy()
            Yp_np = batch.get("Y_pitched")
            Yd_np = batch.get("Y_drum")

            for i in range(len(batch["item_id"])):
                prof = profiles_np.get(batch["style_tgt"][i])
                if prof is None:
                    if Yp_np is None:
                        continue
                    prof = per_item_profile_from_Y(
                        Yp_np[i].numpy(), Yd_np[i].numpy()
                    )
                X_roll = combined_roll_from_input(X_np[i])
                s = score_item_from_rolls(X_roll, pitched_np[i], drum_np[i], prof)
                scores.append(s)
                n += 1
                if n >= max_items:
                    break
            if n >= max_items:
                break

    model.train()
    if not scores:
        return {"CP": 0.0, "SF": 0.0, "score": 0.0, "n": 0}
    return {
        "CP":    float(np.mean([s["CP"]    for s in scores])),
        "SF":    float(np.mean([s["SF"]    for s in scores])),
        "score": float(np.mean([s["score"] for s in scores])),
        "n":     len(scores),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument("--logdir",       type=str, default=None)
    ap.add_argument("--batch-size",   type=int, default=None)
    ap.add_argument("--lr",           type=float, default=None)
    ap.add_argument("--max-steps",    type=int, default=None)
    ap.add_argument("--num-workers",  type=int, default=None)
    ap.add_argument("--resume",       type=str, default=None)
    ap.add_argument("--debug",        action="store_true")
    args = ap.parse_args()

    cfg = Config()
    if args.dataset_root:  cfg.dataset_root = args.dataset_root
    if args.logdir:        cfg.logdir        = args.logdir
    if args.batch_size:    cfg.batch_size    = args.batch_size
    if args.lr:            cfg.lr            = args.lr
    if args.max_steps:     cfg.max_steps     = args.max_steps
    if args.num_workers is not None: cfg.num_workers = args.num_workers
    if args.debug:
        cfg.batch_size  = 2
        cfg.max_steps   = 50
        cfg.log_every   = 5
        cfg.val_every   = 25
        cfg.ckpt_every  = 25

    set_seed(cfg.seed)
    os.makedirs(cfg.logdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOGGER.info(f"Using device: {device}")

    _LOGGER.info("Loading style profiles...")
    profiles_tensor = load_all_profiles(cfg.dataset_root)
    profiles_np     = load_style_profiles_np(cfg.dataset_root)
    _LOGGER.info(f"Loaded {len(profiles_tensor)} style profile tensors, "
                 f"{len(profiles_np)} numpy dicts for scoring.")

    triplets = read_manifest(str(Path(cfg.dataset_root) / "manifest.csv"))
    splits   = split_triplets(triplets,
                              train_split=cfg.train_split,
                              val_split=cfg.val_split,
                              test_split=cfg.test_split)
    train_t, val_t = splits["train"], splits["val"]

    val_check = HackathonRollDataset(
        cfg.dataset_root, val_t, cfg.num_pitches, cfg.num_time_steps,
        require_Y=True, profiles=profiles_tensor,
    )
    if len(val_check) == 0:
        _LOGGER.warning("No val items on disk; carving %.1f%% from train.",
                        cfg.val_from_train_fraction * 100)
        train_t, val_t = carve_val_from_train(
            cfg.dataset_root, train_t, cfg.val_from_train_fraction, cfg.seed
        )

    train_ds = HackathonRollDataset(
        cfg.dataset_root, train_t, cfg.num_pitches, cfg.num_time_steps,
        require_Y=True, profiles=profiles_tensor,
    )
    val_ds = HackathonRollDataset(
        cfg.dataset_root, val_t, cfg.num_pitches, cfg.num_time_steps,
        require_Y=True, profiles=profiles_tensor,
    )
    _LOGGER.info(f"Train: {len(train_ds)} items; val: {len(val_ds)} items")

    if len(train_ds) == 0:
        raise RuntimeError("No train items on disk. Re-download the dataset.")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=max(0, cfg.num_workers - 1), collate_fn=collate,
    )

    model = G2GModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    _LOGGER.info(f"G2GModel parameters: {n_params:,}")

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.max_steps)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        start_step = ck["step"]
        _LOGGER.info(f"Resumed from {args.resume} at step {start_step}")

    step        = start_step
    skipped_nan = 0
    model.train()

    while step < cfg.max_steps:
        for batch in train_loader:
            if step >= cfg.max_steps:
                break

            X            = batch["X"].to(device, non_blocking=True)
            profile_flat = batch["style_flat"].to(device, non_blocking=True)
            Yp           = batch["Y_pitched"].to(device, non_blocking=True)
            Yd           = batch["Y_drum"].to(device, non_blocking=True)

            p_logits, d_logits = model(X, profile_flat)
            loss = (F.binary_cross_entropy_with_logits(p_logits, Yp)
                    + F.binary_cross_entropy_with_logits(d_logits, Yd)) * 0.5

            if not torch.isfinite(loss):
                skipped_nan += 1
                optim.zero_grad(set_to_none=True)
                _LOGGER.warning(f"step {step}: non-finite loss, skipping (total={skipped_nan})")
                step += 1
                continue

            optim.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if not torch.isfinite(gnorm):
                skipped_nan += 1
                optim.zero_grad(set_to_none=True)
                _LOGGER.warning(f"step {step}: non-finite gnorm, skipping (total={skipped_nan})")
                step += 1
                continue

            optim.step()
            sched.step()
            step += 1

            if step % cfg.log_every == 0:
                _LOGGER.info(
                    f"step {step:7d}  loss {loss.item():.4f}  "
                    f"gnorm {gnorm.item():.2f}  "
                    f"lr {optim.param_groups[0]['lr']:.2e}"
                )

            if step % cfg.val_every == 0 and len(val_ds) > 0:
                m = evaluate(model, val_loader, device, profiles_np)
                _LOGGER.info(
                    f"VAL step {step}  n={m['n']}  CP={m['CP']:.3f}  "
                    f"SF={m['SF']:.3f}  HM={m['score']:.3f}"
                )

            if step % cfg.ckpt_every == 0 or step == cfg.max_steps:
                ckpt_dir = Path(cfg.logdir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "sched": sched.state_dict(),
                    "step":  step,
                    "cfg":   cfg.__dict__,
                }
                tagged = ckpt_dir / f"model_{step:07d}.pt"
                alias  = ckpt_dir / cfg.ckpt_name
                torch.save(payload, tagged)
                torch.save(payload, alias)
                _LOGGER.info(f"Saved {tagged} (+ alias {alias.name})")

    _LOGGER.info(f"Training done. Skipped {skipped_nan} non-finite batches.")


if __name__ == "__main__":
    main()
