"""Train the PyTorch Groove2Groove port.

Usage:
    cd code
    python -m g2g_pytorch.train --dataset-root ../dataset --logdir runs/exp1

The script reads ``manifest.csv``, builds train/val DataLoaders, runs the
``G2GModel`` against weighted BCE-with-logits on the Y pitched + drum
rolls, periodically evaluates the harmonic-mean score on val (using the
supplied ``utility.metric``) and writes checkpoints to ``--logdir``.
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

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .config import Config
from .dataset import (
    HackathonRollDataset,
    carve_val_from_train,
    collate,
    read_manifest,
    split_triplets,
)
from .models import G2GModel
from .metrics import (
    combined_roll_from_input,
    score_item_from_rolls,
    torch_logits_to_rolls,
    per_item_profile_from_Y,
)


_LOGGER = logging.getLogger("g2g_pytorch.train")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_bce_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """BCE-with-logits on soft targets in [0, 1] with a single positive weight."""
    w = torch.full((1,), float(pos_weight), device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=w)


def evaluate(model: G2GModel, loader: DataLoader, cfg: Config, device: torch.device,
             style_profiles: Dict[str, Dict[str, np.ndarray]], max_items: int = 200) -> Dict[str, float]:
    model.eval()
    scores: List[Dict[str, float]] = []
    n = 0
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            Z = batch["Z"].to(device)
            lp, ld = model(X, Z)
            p, d = torch_logits_to_rolls(lp, ld, cfg.velocity_threshold)
            X_np = batch["X"].numpy()
            Yp_np = batch["Y_pitched"].numpy() if "Y_pitched" in batch else None
            Yd_np = batch["Y_drum"].numpy() if "Y_drum" in batch else None
            for i, iid in enumerate(batch["item_id"]):
                prof = style_profiles.get(batch["style_tgt"][i])
                if prof is None:
                    # Fallback for items whose target style has no precomputed
                    # profile (e.g. val carved out of train).
                    if Yp_np is None:
                        continue
                    prof = per_item_profile_from_Y(Yp_np[i], Yd_np[i])
                X_roll = combined_roll_from_input(X_np[i])
                s = score_item_from_rolls(X_roll, p[i], d[i], prof)
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
        "CP": float(np.mean([s["CP"] for s in scores])),
        "SF": float(np.mean([s["SF"] for s in scores])),
        "score": float(np.mean([s["score"] for s in scores])),
        "n": len(scores),
    }


def load_style_profiles(root: str) -> Dict[str, Dict[str, np.ndarray]]:
    profiles_dir = Path(root) / "style_profiles"
    profiles: Dict[str, Dict[str, np.ndarray]] = {}
    if not profiles_dir.exists():
        return profiles
    for f in profiles_dir.iterdir():
        if f.suffix != ".npz":
            continue
        with np.load(f) as z:
            profiles[f.stem] = {k: z[k] for k in z.files}
    return profiles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, default=None,
                    help="Path to the dataset/ directory (with manifest.csv).")
    ap.add_argument("--logdir", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from.")
    ap.add_argument("--debug", action="store_true",
                    help="Tiny config for smoke-testing.")
    args = ap.parse_args()

    cfg = Config()
    if args.dataset_root: cfg.dataset_root = args.dataset_root
    if args.logdir: cfg.logdir = args.logdir
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr: cfg.lr = args.lr
    if args.max_steps: cfg.max_steps = args.max_steps
    if args.num_workers is not None: cfg.num_workers = args.num_workers
    if args.debug:
        cfg.batch_size = 4
        cfg.max_steps = 50
        cfg.log_every = 5
        cfg.val_every = 25
        cfg.ckpt_every = 25
        cfg.decoder_layers = 1
        cfg.decoder_hidden = 128
        cfg.content_rnn_hidden = 64
        cfg.style_rnn_hidden = 64
        cfg.style_dim = 64

    set_seed(cfg.seed)
    os.makedirs(cfg.logdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOGGER.info(f"Using device: {device}")

    triplets = read_manifest(str(Path(cfg.dataset_root) / "manifest.csv"))
    splits = split_triplets(triplets,
                            train_split=cfg.train_split,
                            val_split=cfg.val_split,
                            test_split=cfg.test_split)
    train_t, val_t = splits["train"], splits["val"]

    # If no val NPZs are on disk, hold out a slice of train.
    val_dataset_first_pass = HackathonRollDataset(
        cfg.dataset_root, val_t, cfg.num_pitches, cfg.num_time_steps, require_Y=True
    )
    if len(val_dataset_first_pass) == 0:
        _LOGGER.warning(
            "No val items found on disk; carving %.1f%% out of train as held-out.",
            cfg.val_from_train_fraction * 100,
        )
        train_t, val_t = carve_val_from_train(
            cfg.dataset_root, train_t, cfg.val_from_train_fraction, cfg.seed
        )

    train_ds = HackathonRollDataset(
        cfg.dataset_root, train_t, cfg.num_pitches, cfg.num_time_steps, require_Y=True
    )
    val_ds = HackathonRollDataset(
        cfg.dataset_root, val_t, cfg.num_pitches, cfg.num_time_steps, require_Y=True
    )
    _LOGGER.info(
        f"Train items: {len(train_ds)} (skipped {train_ds._skipped} missing); "
        f"val items: {len(val_ds)} (skipped {val_ds._skipped} missing)"
    )

    if len(train_ds) == 0:
        raise RuntimeError(
            "No train items present on disk. Re-download the dataset first."
        )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=max(0, cfg.num_workers - 1), collate_fn=collate,
    )

    style_profiles = load_style_profiles(cfg.dataset_root)
    _LOGGER.info(f"Loaded {len(style_profiles)} style profiles.")

    model = G2GModel(cfg).to(device)
    # Trigger lazy-build inside encoders.
    with torch.no_grad():
        dummy_x = torch.zeros(1, cfg.in_channels, cfg.num_pitches, cfg.num_time_steps, device=device)
        model(dummy_x, dummy_x)
    n_params = sum(p.numel() for p in model.parameters())
    _LOGGER.info(f"Model parameters: {n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.max_steps)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        start_step = ck["step"]
        _LOGGER.info(f"Resumed from {args.resume} at step {start_step}")

    step = start_step
    skipped_nan = 0
    model.train()
    while step < cfg.max_steps:
        for batch in train_loader:
            if step >= cfg.max_steps:
                break
            X = batch["X"].to(device, non_blocking=True)
            Z = batch["Z"].to(device, non_blocking=True)
            Yp = batch["Y_pitched"].to(device, non_blocking=True)
            Yd = batch["Y_drum"].to(device, non_blocking=True)

            logits_p, logits_d = model(X, Z)
            loss_p = soft_bce_loss(logits_p, Yp, cfg.pos_weight)
            loss_d = soft_bce_loss(logits_d, Yd, cfg.pos_weight)

            # Histogram loss: soft SF metric directly on the output.
            pred_p = torch.sigmoid(logits_p)   # (B, 128, 128)
            pred_d = torch.sigmoid(logits_d)   # (B, 128, 128)
            # Pitch density — average velocity per pitch over time
            loss_h_pitch = 1 - F.cosine_similarity(
                pred_p.mean(dim=-1), Yp.mean(dim=-1).detach(), dim=-1).mean()
            # Onset activity — average pitched activity per time step
            loss_h_onset = 1 - F.cosine_similarity(
                pred_p.mean(dim=1), Yp.mean(dim=1).detach(), dim=-1).mean()
            # Drum rhythm — average drum activity per time step
            loss_h_drum = 1 - F.cosine_similarity(
                pred_d.mean(dim=1), Yd.mean(dim=1).detach(), dim=-1).mean()
            loss_hist = (loss_h_pitch + loss_h_onset + loss_h_drum) / 3

            loss = (cfg.pitched_loss_weight * loss_p
                    + cfg.drum_loss_weight   * loss_d
                    + cfg.hist_loss_weight   * loss_hist)

            # NaN guard: a single bad batch corrupts every parameter for the rest
            # of training. Drop it on the floor instead.
            if not torch.isfinite(loss):
                skipped_nan += 1
                optim.zero_grad(set_to_none=True)
                _LOGGER.warning(
                    f"step {step}: non-finite loss (p={loss_p.item():.3g} "
                    f"d={loss_d.item():.3g}), skipping batch "
                    f"(total skipped={skipped_nan})"
                )
                step += 1
                continue

            optim.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            # Belt-and-suspenders: if the gradient itself is non-finite, the
            # optimizer step would propagate NaN to every parameter. Skip.
            if not torch.isfinite(gnorm):
                skipped_nan += 1
                optim.zero_grad(set_to_none=True)
                _LOGGER.warning(
                    f"step {step}: non-finite grad norm, skipping batch "
                    f"(total skipped={skipped_nan})"
                )
                step += 1
                continue
            optim.step()
            sched.step()
            step += 1

            if step % cfg.log_every == 0:
                _LOGGER.info(
                    f"step {step:7d}  loss {loss.item():.4f}  "
                    f"p {loss_p.item():.4f}  d {loss_d.item():.4f}  "
                    f"hist {loss_hist.item():.4f}  "
                    f"gnorm {gnorm.item():.2f}  "
                    f"lr {optim.param_groups[0]['lr']:.2e}"
                )
            if step % cfg.val_every == 0 and len(val_ds) > 0 and style_profiles:
                m = evaluate(model, val_loader, cfg, device, style_profiles)
                _LOGGER.info(
                    f"VAL step {step}  n={m['n']}  CP={m['CP']:.3f}  "
                    f"SF={m['SF']:.3f}  HM={m['score']:.3f}"
                )
            if step % cfg.ckpt_every == 0 or step == cfg.max_steps:
                # Write a step-tagged copy AND update the rolling 'model.pt'
                # alias. The tagged copies survive a future NaN; the alias is
                # what `--resume model.pt` and `infer --ckpt model.pt` expect.
                ckpt_dir = Path(cfg.logdir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "sched": sched.state_dict(),
                    "step": step,
                    "cfg": cfg.__dict__,
                }
                tagged = ckpt_dir / f"model_{step:07d}.pt"
                alias = ckpt_dir / cfg.ckpt_name
                torch.save(payload, tagged)
                torch.save(payload, alias)
                _LOGGER.info(f"Saved checkpoint to {tagged} (+ alias {alias.name})")

    _LOGGER.info(f"Training done. Skipped {skipped_nan} non-finite batches.")


if __name__ == "__main__":
    main()
