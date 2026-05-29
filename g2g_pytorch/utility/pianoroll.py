from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pretty_midi

    _HAS_PRETTY_MIDI = True
except Exception:
    pretty_midi = None
    _HAS_PRETTY_MIDI = False


NUM_PITCHES = 128
STEPS_PER_BEAT = 4
BEATS_PER_BAR = 4
BARS_PER_FRAGMENT = 8
T_PER_FRAGMENT = STEPS_PER_BEAT * BEATS_PER_BAR * BARS_PER_FRAGMENT

VELOCITY_THRESHOLD = 0.05

MIN_NOTE_CELLS = 1


@dataclass(frozen=True)
class Note:
    pitch: int
    onset_beats: float
    duration_beats: float
    velocity: int
    is_drum: bool = False
    track_id: int = 0


def beats_to_steps(beats: float) -> int:
    return int(round(beats * STEPS_PER_BEAT))


def steps_to_beats(steps: int) -> float:
    return float(steps) / STEPS_PER_BEAT


def notes_to_roll(
    notes: Iterable[Note],
    num_steps: int = T_PER_FRAGMENT,
    drums: bool = False,
) -> np.ndarray:
    roll = np.zeros((NUM_PITCHES, num_steps), dtype=np.float32)
    for n in notes:
        if bool(n.is_drum) != bool(drums):
            continue
        start = beats_to_steps(n.onset_beats)
        end = start + max(beats_to_steps(n.duration_beats), MIN_NOTE_CELLS)
        if start >= num_steps:
            continue
        end = min(end, num_steps)
        v = float(np.clip(n.velocity, 1, 127)) / 127.0
        roll[int(n.pitch), start:end] = np.maximum(roll[int(n.pitch), start:end], v)
    return roll


def roll_to_notes(
    roll: np.ndarray,
    threshold: float = VELOCITY_THRESHOLD,
    is_drum: bool = False,
    track_id: int = 0,
) -> List[Note]:
    if roll.ndim != 2 or roll.shape[0] != NUM_PITCHES:
        raise ValueError(f"roll must have shape (128, T); got {roll.shape}")
    active = roll > threshold
    notes: List[Note] = []
    T = roll.shape[1]
    for pitch in range(NUM_PITCHES):
        row = active[pitch]
        if not row.any():
            continue
        diff = np.diff(row.astype(np.int8), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if e <= s:
                continue
            vmax = float(roll[pitch, s:e].max())
            vel = int(np.clip(round(vmax * 127.0), 1, 127))
            notes.append(
                Note(
                    pitch=int(pitch),
                    onset_beats=steps_to_beats(int(s)),
                    duration_beats=steps_to_beats(int(e - s)),
                    velocity=vel,
                    is_drum=bool(is_drum),
                    track_id=int(track_id),
                )
            )
    return notes


def midi_to_notes(
    midi_path: str,
    num_beats: Optional[int] = None,
) -> List[Note]:
    if not _HAS_PRETTY_MIDI:
        raise RuntimeError(
            "pretty_midi is required for midi_to_notes(); pip install pretty_midi"
        )
    pm = pretty_midi.PrettyMIDI(midi_path)
    tempo = pm.estimate_tempo() if pm.instruments else 120.0
    sec_per_beat = 60.0 / max(tempo, 1e-3)
    out: List[Note] = []
    for ti, inst in enumerate(pm.instruments):
        for n in inst.notes:
            onset_beats = n.start / sec_per_beat
            dur_beats = max(0.0, (n.end - n.start) / sec_per_beat)
            ob = round(onset_beats * STEPS_PER_BEAT) / STEPS_PER_BEAT
            db = max(
                round(dur_beats * STEPS_PER_BEAT) / STEPS_PER_BEAT,
                1.0 / STEPS_PER_BEAT,
            )
            if num_beats is not None and ob >= num_beats:
                continue
            out.append(
                Note(
                    pitch=int(n.pitch),
                    onset_beats=float(ob),
                    duration_beats=float(db),
                    velocity=int(np.clip(n.velocity, 1, 127)),
                    is_drum=bool(inst.is_drum),
                    track_id=int(ti),
                )
            )
    return out


def notes_to_midi(
    notes: Sequence[Note],
    output_path: str,
    tempo: float = 120.0,
) -> None:
    if not _HAS_PRETTY_MIDI:
        raise RuntimeError(
            "pretty_midi is required for notes_to_midi(); pip install pretty_midi"
        )
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    sec_per_beat = 60.0 / max(tempo, 1e-3)
    groups: dict = {}
    for n in notes:
        key = (int(n.track_id), bool(n.is_drum))
        groups.setdefault(key, []).append(n)
    for (track_id, is_drum), group in groups.items():
        program = 0 if not is_drum else 0
        inst = pretty_midi.Instrument(
            program=program, is_drum=is_drum, name=f"track{track_id}"
        )
        for n in group:
            start = n.onset_beats * sec_per_beat
            end = start + max(n.duration_beats * sec_per_beat, 1e-3)
            inst.notes.append(
                pretty_midi.Note(
                    velocity=int(np.clip(n.velocity, 1, 127)),
                    pitch=int(np.clip(n.pitch, 0, 127)),
                    start=float(start),
                    end=float(end),
                )
            )
        pm.instruments.append(inst)
    pm.write(output_path)


def combined_roll(notes: Iterable[Note], num_steps: int = T_PER_FRAGMENT) -> np.ndarray:
    roll = np.zeros((NUM_PITCHES, num_steps), dtype=np.float32)
    for n in notes:
        start = beats_to_steps(n.onset_beats)
        end = start + max(beats_to_steps(n.duration_beats), MIN_NOTE_CELLS)
        if start >= num_steps:
            continue
        end = min(end, num_steps)
        v = float(np.clip(n.velocity, 1, 127)) / 127.0
        roll[int(n.pitch), start:end] = np.maximum(roll[int(n.pitch), start:end], v)
    return roll


def chroma_from_roll(roll: np.ndarray) -> np.ndarray:
    if roll.shape[0] != NUM_PITCHES:
        raise ValueError("expected 128 pitches on the first axis")
    chroma = np.zeros((12, roll.shape[1]), dtype=np.float32)
    for p in range(NUM_PITCHES):
        chroma[p % 12] += roll[p]
    return chroma


def windowed_chroma(
    roll: np.ndarray,
    frames_per_beat: int = 12,
    window_beats: float = 2.0,
    stride_beats: float = 1.0,
) -> np.ndarray:
    chroma = chroma_from_roll(roll)
    if frames_per_beat != STEPS_PER_BEAT:
        repeat = max(1, int(round(frames_per_beat / STEPS_PER_BEAT)))
        chroma = np.repeat(chroma, repeat, axis=1)
        frames_per_beat = STEPS_PER_BEAT * repeat
    window = max(1, int(round(window_beats * frames_per_beat)))
    stride = max(1, int(round(stride_beats * frames_per_beat)))
    T = chroma.shape[1]
    if T < window:
        return chroma.mean(axis=1, keepdims=True)
    starts = list(range(0, T - window + 1, stride))
    out = np.stack([chroma[:, s : s + window].mean(axis=1) for s in starts], axis=1)
    return out


__all__ = [
    "NUM_PITCHES",
    "STEPS_PER_BEAT",
    "BEATS_PER_BAR",
    "BARS_PER_FRAGMENT",
    "T_PER_FRAGMENT",
    "VELOCITY_THRESHOLD",
    "Note",
    "beats_to_steps",
    "steps_to_beats",
    "notes_to_roll",
    "roll_to_notes",
    "midi_to_notes",
    "notes_to_midi",
    "combined_roll",
    "chroma_from_roll",
    "windowed_chroma",
]
