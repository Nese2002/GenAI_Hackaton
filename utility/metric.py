from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import pianoroll as pr


TIME_BINS = pr.BARS_PER_FRAGMENT * pr.BEATS_PER_BAR
PITCH_BINS = pr.NUM_PITCHES
DURATION_BINS = 16
VELOCITY_BINS = 8
ONSET_BINS = pr.BEATS_PER_BAR * pr.STEPS_PER_BEAT
DRUM_BINS = pr.NUM_PITCHES

EPS = 1e-12


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def content_preservation(
    roll_X: np.ndarray,
    roll_out: np.ndarray,
    frames_per_beat: int = 12,
    window_beats: float = 2.0,
    stride_beats: float = 1.0,
) -> float:
    cX = pr.windowed_chroma(roll_X, frames_per_beat, window_beats, stride_beats)
    cO = pr.windowed_chroma(roll_out, frames_per_beat, window_beats, stride_beats)
    n = min(cX.shape[1], cO.shape[1])
    if n == 0:
        return 0.0
    sims = []
    for i in range(n):
        sims.append(_cosine(cX[:, i], cO[:, i]))
    return float(np.mean(sims))


def _bundle_to_notes(bundle: Dict[str, np.ndarray]) -> List[pr.Note]:
    notes: List[pr.Note] = []
    pitched = bundle.get("pitched")
    drum = bundle.get("drum")
    if pitched is not None and pitched.size > 0:
        for i in range(pitched.shape[0]):
            notes.extend(pr.roll_to_notes(pitched[i], is_drum=False, track_id=int(i)))
    if drum is not None and drum.size > 0:
        notes.extend(pr.roll_to_notes(drum, is_drum=True, track_id=999))
    return notes


def _time_pitch_hist(notes: Iterable[pr.Note]) -> np.ndarray:
    h = np.zeros((TIME_BINS, PITCH_BINS), dtype=np.float64)
    for n in notes:
        if n.is_drum:
            continue
        t = int(np.clip(int(n.onset_beats), 0, TIME_BINS - 1))
        p = int(np.clip(n.pitch, 0, PITCH_BINS - 1))
        h[t, p] += 1.0
    return h


def _onset_duration_hist(notes: Iterable[pr.Note]) -> np.ndarray:
    h = np.zeros((ONSET_BINS, DURATION_BINS), dtype=np.float64)
    for n in notes:
        if n.is_drum:
            continue
        onset_cell = int(round(n.onset_beats * pr.STEPS_PER_BEAT)) % ONSET_BINS
        dur_cell = int(
            np.clip(
                int(round(n.duration_beats * pr.STEPS_PER_BEAT)) - 1,
                0,
                DURATION_BINS - 1,
            )
        )
        h[onset_cell, dur_cell] += 1.0
    return h


def _onset_velocity_hist(notes: Iterable[pr.Note]) -> np.ndarray:
    h = np.zeros((ONSET_BINS, VELOCITY_BINS), dtype=np.float64)
    for n in notes:
        if n.is_drum:
            continue
        onset_cell = int(round(n.onset_beats * pr.STEPS_PER_BEAT)) % ONSET_BINS
        v = int(np.clip(int(n.velocity * VELOCITY_BINS / 128), 0, VELOCITY_BINS - 1))
        h[onset_cell, v] += 1.0
    return h


def _onset_drum_hist(notes: Iterable[pr.Note]) -> np.ndarray:
    h = np.zeros((ONSET_BINS, DRUM_BINS), dtype=np.float64)
    for n in notes:
        if not n.is_drum:
            continue
        onset_cell = int(round(n.onset_beats * pr.STEPS_PER_BEAT)) % ONSET_BINS
        p = int(np.clip(n.pitch, 0, DRUM_BINS - 1))
        h[onset_cell, p] += 1.0
    return h


HIST_NAMES = ("time_pitch", "onset_duration", "onset_velocity", "onset_drum")


def compute_histograms(bundle_or_notes) -> Dict[str, np.ndarray]:
    if isinstance(bundle_or_notes, dict):
        notes = _bundle_to_notes(bundle_or_notes)
    else:
        notes = list(bundle_or_notes)
    return {
        "time_pitch": _time_pitch_hist(notes),
        "onset_duration": _onset_duration_hist(notes),
        "onset_velocity": _onset_velocity_hist(notes),
        "onset_drum": _onset_drum_hist(notes),
    }


def build_style_profile_from_bundles(
    bundles: Sequence[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    acc: Optional[Dict[str, np.ndarray]] = None
    for b in bundles:
        hs = compute_histograms(b)
        if acc is None:
            acc = {k: v.copy() for k, v in hs.items()}
        else:
            for k in acc:
                acc[k] += hs[k]
    if acc is None:
        return {
            "time_pitch": np.zeros((TIME_BINS, PITCH_BINS), dtype=np.float64),
            "onset_duration": np.zeros((ONSET_BINS, DURATION_BINS), dtype=np.float64),
            "onset_velocity": np.zeros((ONSET_BINS, VELOCITY_BINS), dtype=np.float64),
            "onset_drum": np.zeros((ONSET_BINS, DRUM_BINS), dtype=np.float64),
        }
    return acc


def style_fit(
    output_bundle_or_notes,
    target_profile: Dict[str, np.ndarray],
) -> float:
    out_hists = compute_histograms(output_bundle_or_notes)
    sims = []
    for k in HIST_NAMES:
        if k not in target_profile:
            continue
        sims.append(_cosine(out_hists[k], target_profile[k]))
    if not sims:
        return 0.0
    return float(np.mean(sims))


def harmonic_mean(cp: float, sf: float) -> float:
    s = float(cp) + float(sf)
    if s <= 0.0:
        return 0.0
    return 2.0 * float(cp) * float(sf) / s


def score_item(
    content_roll: np.ndarray,
    output_bundle: Dict[str, np.ndarray],
    target_profile: Dict[str, np.ndarray],
) -> Dict[str, float]:
    from .data import combined_roll_from_bundle

    out_roll = combined_roll_from_bundle(output_bundle)
    cp = content_preservation(content_roll, out_roll)
    sf = style_fit(output_bundle, target_profile)
    return {"CP": float(cp), "SF": float(sf), "score": harmonic_mean(cp, sf)}


def leaderboard_score(per_item: Sequence[Dict[str, float]]) -> float:
    if not per_item:
        return 0.0
    return float(np.mean([d["score"] for d in per_item]))


__all__ = [
    "TIME_BINS",
    "PITCH_BINS",
    "DURATION_BINS",
    "VELOCITY_BINS",
    "ONSET_BINS",
    "DRUM_BINS",
    "HIST_NAMES",
    "content_preservation",
    "compute_histograms",
    "build_style_profile_from_bundles",
    "style_fit",
    "harmonic_mean",
    "score_item",
    "leaderboard_score",
]
