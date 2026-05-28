"""Submission writer + ``validate_submission()`` for the hackathon.

Kaggle Community Competitions are row-keyed: the submission CSV must
have **one row per ID** so the platform can join it against the
solution file. We therefore use a single ``notes`` column per item:

    ID, notes

* ``ID``    — string ``item_id`` (must appear in the test manifest).
* ``notes`` — semicolon-separated note records, each of the form

      track_id,is_drum,pitch,onset_steps,dur_steps,velocity

  Six comma-separated integers. ``onset_steps`` and ``dur_steps`` are
  counts of 16th-note cells (``STEPS_PER_BEAT = 4``). ``track_id`` >= 0
  (``999`` is conventionally the drum track), ``is_drum`` in {0, 1},
  ``pitch`` in 0..127, ``dur_steps`` >= 1, ``velocity`` in 1..127.

Every ``item_id`` in the test set must appear in exactly one row. The
``notes`` cell may be empty (zero notes for that item) but the row must
exist — the scorer reconstructs piano-rolls from these notes, so any
missing item is rejected.

The intermediate, in-memory representation used by the baselines and
the student template is still a per-note dict::

    {item_id, track_id, is_drum, pitch, onset_beats, duration_beats, velocity}

:func:`notes_to_rows` / :func:`bundle_to_rows` yield these dicts;
:func:`write_submission` aggregates them into the wire format;
:func:`validate_submission` decodes the wire format back to per-note
rows so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except Exception:  # pragma: no cover
    pd = None  # type: ignore
    _HAS_PANDAS = False

from . import pianoroll as pr


# Public wire-format columns (what Kaggle sees).
SUBMISSION_COLUMNS: Tuple[str, ...] = ("ID", "notes")

# Internal per-note columns (intermediate representation in baselines /
# template, and the DataFrame returned by ``validate_submission``).
PER_NOTE_COLUMNS: Tuple[str, ...] = (
    "item_id",
    "track_id",
    "is_drum",
    "pitch",
    "onset_beats",
    "duration_beats",
    "velocity",
)


class SubmissionError(ValueError):
    """Raised when a submission is malformed."""


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _ensure_pandas():
    if not _HAS_PANDAS:
        raise RuntimeError("pandas is required for utility.submission")


def notes_to_rows(item_id: str, notes: Iterable[pr.Note]):
    """Yield per-note dicts for one item (intermediate representation)."""
    for n in notes:
        yield {
            "item_id": str(item_id),
            "track_id": int(n.track_id),
            "is_drum": int(bool(n.is_drum)),
            "pitch": int(np.clip(n.pitch, 0, 127)),
            "onset_beats": float(
                round(n.onset_beats * pr.STEPS_PER_BEAT) / pr.STEPS_PER_BEAT
            ),
            "duration_beats": float(
                max(
                    1.0 / pr.STEPS_PER_BEAT,
                    round(n.duration_beats * pr.STEPS_PER_BEAT) / pr.STEPS_PER_BEAT,
                )
            ),
            "velocity": int(np.clip(n.velocity, 1, 127)),
        }


def bundle_to_rows(item_id: str, bundle):
    """Convert a roll bundle (pitched + drum) into per-note dicts."""
    pitched = bundle.get("pitched")
    drum = bundle.get("drum")
    track_ids = bundle.get("track_ids")
    notes: List[pr.Note] = []
    if pitched is not None and pitched.size > 0:
        for i in range(pitched.shape[0]):
            tid = int(track_ids[i]) if track_ids is not None else int(i)
            notes.extend(pr.roll_to_notes(pitched[i], is_drum=False, track_id=tid))
    if drum is not None and drum.size > 0:
        notes.extend(pr.roll_to_notes(drum, is_drum=True, track_id=999))
    yield from notes_to_rows(item_id, notes)


# ---------------------------------------------------------------------------
# Encoding / decoding of the ``notes`` cell.
# ---------------------------------------------------------------------------


def _encode_one(row: dict) -> str:
    onset_steps = int(round(float(row["onset_beats"]) * pr.STEPS_PER_BEAT))
    dur_steps = int(round(float(row["duration_beats"]) * pr.STEPS_PER_BEAT))
    if dur_steps < 1:
        dur_steps = 1
    return (
        f"{int(row['track_id'])},{int(row['is_drum'])},{int(row['pitch'])},"
        f"{onset_steps},{dur_steps},{int(row['velocity'])}"
    )


def encode_notes(rows: Iterable[dict]) -> str:
    """Encode per-note dicts into the ``notes`` cell string."""
    return ";".join(_encode_one(r) for r in rows)


def _decode_one(item_id: str, token: str) -> dict:
    parts = token.split(",")
    if len(parts) != 6:
        raise SubmissionError(
            f"item_id={item_id!r}: bad note record {token!r} "
            "(expected 'track_id,is_drum,pitch,onset_steps,dur_steps,velocity')"
        )
    try:
        track_id, is_drum, pitch, onset_steps, dur_steps, velocity = (
            int(x) for x in parts
        )
    except ValueError as e:
        raise SubmissionError(
            f"item_id={item_id!r}: cannot parse note {token!r}: {e}"
        ) from e
    return {
        "item_id": str(item_id),
        "track_id": track_id,
        "is_drum": is_drum,
        "pitch": pitch,
        "onset_beats": float(onset_steps) / pr.STEPS_PER_BEAT,
        "duration_beats": float(dur_steps) / pr.STEPS_PER_BEAT,
        "velocity": velocity,
    }


def decode_notes(item_id: str, notes_str) -> List[dict]:
    """Decode a ``notes`` cell into a list of per-note dicts (possibly empty)."""
    if notes_str is None:
        return []
    if not isinstance(notes_str, str):
        try:
            if np.isnan(float(notes_str)):
                return []
        except (TypeError, ValueError):
            pass
        notes_str = str(notes_str)
    s = notes_str.strip()
    if not s:
        return []
    return [_decode_one(item_id, tok) for tok in s.split(";") if tok]


# ---------------------------------------------------------------------------
# Writer.
# ---------------------------------------------------------------------------


def write_submission(
    output_path: str,
    rows: Iterable[dict],
) -> str:
    """Write per-note rows to a Kaggle-format submission CSV.

    ``rows`` is the per-note dict stream produced by :func:`notes_to_rows`
    or :func:`bundle_to_rows`. We group rows by ``item_id`` and emit one
    row per item with a serialized ``notes`` column (the format Kaggle
    expects). The item ordering of the first occurrence of each id is
    preserved so the CSV is deterministic.
    """
    _ensure_pandas()
    rows_list = list(rows)
    if not rows_list:
        raise SubmissionError(
            "no rows to write — every item_id must produce at least one note"
        )
    by_item: Dict[str, List[dict]] = {}
    order: List[str] = []
    for r in rows_list:
        iid = str(r["item_id"])
        if iid not in by_item:
            by_item[iid] = []
            order.append(iid)
        by_item[iid].append(r)
    out_rows = [{"ID": iid, "notes": encode_notes(by_item[iid])} for iid in order]
    df = pd.DataFrame(out_rows, columns=list(SUBMISSION_COLUMNS))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


# ---------------------------------------------------------------------------
# Validator.
# ---------------------------------------------------------------------------


def validate_submission(
    submission: Union[str, Path, "pd.DataFrame"],
    required_item_ids: Optional[Sequence[str]] = None,
) -> "pd.DataFrame":
    """Validate a submission and return a per-note DataFrame.

    Accepts either:
      * a path / DataFrame in the **wire format** (``ID, notes``) — the
        Kaggle submission format; this is the common case.
      * a DataFrame already in the **per-note format**
        (``item_id, track_id, ..., velocity``) — convenient when callers
        have an in-memory DataFrame they want to round-trip-check.

    Checks performed:
      * required columns present.
      * every ``ID`` is unique (Kaggle requires one row per id).
      * value ranges (pitch 0–127, velocity 1–127, on-grid onsets/durs,
        ``is_drum`` in {0, 1}).
      * every entry in ``required_item_ids`` is covered.

    Returns a per-note DataFrame compatible with :func:`rows_to_bundles`.
    Raises :class:`SubmissionError` on the first violation.
    """
    _ensure_pandas()
    if isinstance(submission, (str, Path)):
        try:
            sub = pd.read_csv(str(submission))
        except Exception as e:
            raise SubmissionError(f"cannot read CSV: {e!r}") from e
    elif hasattr(submission, "copy"):
        sub = submission.copy()
    else:
        raise SubmissionError("submission must be a path or a pandas DataFrame")

    has_wire = all(c in sub.columns for c in SUBMISSION_COLUMNS)
    has_per_note = all(c in sub.columns for c in PER_NOTE_COLUMNS)

    if has_wire:
        sub = sub.loc[:, list(SUBMISSION_COLUMNS)].copy()
        sub["ID"] = sub["ID"].astype(str)
        sub["notes"] = sub["notes"].where(sub["notes"].notna(), "").astype(str)
        if sub["ID"].duplicated().any():
            dups = sub.loc[sub["ID"].duplicated(), "ID"].head(5).tolist()
            raise SubmissionError(
                f"duplicate ID rows in submission (first 5: {dups}); "
                "every item must appear in exactly one row"
            )
        per_note_rows: List[dict] = []
        for _, r in sub.iterrows():
            per_note_rows.extend(decode_notes(r["ID"], r["notes"]))
        if per_note_rows:
            df = pd.DataFrame(per_note_rows, columns=list(PER_NOTE_COLUMNS))
        else:
            df = pd.DataFrame({c: pd.Series(dtype=object) for c in PER_NOTE_COLUMNS})
        present_ids = set(sub["ID"].tolist())
    elif has_per_note:
        df = sub.loc[:, list(PER_NOTE_COLUMNS)].copy()
        present_ids = set(df["item_id"].astype(str).unique())
    else:
        missing = [c for c in SUBMISSION_COLUMNS if c not in sub.columns]
        raise SubmissionError(
            f"missing required columns: {missing} "
            f"(found: {list(sub.columns)}). Expected wire format "
            f"{list(SUBMISSION_COLUMNS)}."
        )

    if len(df) > 0:
        try:
            df["item_id"] = df["item_id"].astype(str)
            df["track_id"] = df["track_id"].astype(int)
            df["is_drum"] = df["is_drum"].astype(int)
            df["pitch"] = df["pitch"].astype(int)
            df["onset_beats"] = df["onset_beats"].astype(float)
            df["duration_beats"] = df["duration_beats"].astype(float)
            df["velocity"] = df["velocity"].astype(int)
        except Exception as e:
            raise SubmissionError(f"could not coerce dtypes: {e!r}") from e

        if not df["is_drum"].isin([0, 1]).all():
            raise SubmissionError("is_drum must be 0 or 1")
        if (df["pitch"] < 0).any() or (df["pitch"] > 127).any():
            raise SubmissionError("pitch must lie in [0, 127]")
        if (df["velocity"] < 1).any() or (df["velocity"] > 127).any():
            raise SubmissionError("velocity must lie in [1, 127]")
        if (df["onset_beats"] < 0).any():
            raise SubmissionError("onset_beats must be >= 0")
        if (df["duration_beats"] <= 0).any():
            raise SubmissionError("duration_beats must be > 0")
        if (df["track_id"] < 0).any():
            raise SubmissionError("track_id must be >= 0")

        onset_cells = df["onset_beats"].values * pr.STEPS_PER_BEAT
        dur_cells = df["duration_beats"].values * pr.STEPS_PER_BEAT
        if (np.abs(onset_cells - np.round(onset_cells)) > 1e-3).any():
            raise SubmissionError("onset_beats values are not on the 16th-note grid")
        if (np.abs(dur_cells - np.round(dur_cells)) > 1e-3).any():
            raise SubmissionError("duration_beats values are not on the 16th-note grid")

    if required_item_ids is not None:
        required = set(map(str, required_item_ids))
        missing_ids = sorted(required - present_ids)
        if missing_ids:
            raise SubmissionError(
                f"submission missing item_id(s): first 5 = {missing_ids[:5]} "
                f"(total {len(missing_ids)} missing)"
            )

    return df


# ---------------------------------------------------------------------------
# Group-by helpers used by baselines, the template, and the scorer.
# ---------------------------------------------------------------------------


def rows_to_bundles(df) -> dict:
    """Turn a per-note DataFrame into per-item roll bundles.

    Returns ``{item_id: bundle}``; each bundle has the same layout as
    :func:`utility.data.load_roll_bundle`. Items present in
    ``required_item_ids`` but with zero notes will be missing from the
    returned dict — callers that need a full mapping should handle this.
    """
    _ensure_pandas()
    out: dict = {}
    for item_id, sub in df.groupby("item_id"):
        notes = []
        for _, r in sub.iterrows():
            notes.append(
                pr.Note(
                    pitch=int(r["pitch"]),
                    onset_beats=float(r["onset_beats"]),
                    duration_beats=float(r["duration_beats"]),
                    velocity=int(r["velocity"]),
                    is_drum=bool(int(r["is_drum"])),
                    track_id=int(r["track_id"]),
                )
            )
        pitched_tracks = sorted(
            {
                int(r["track_id"])
                for r in sub.to_dict("records")
                if not int(r["is_drum"])
            }
        )
        if not pitched_tracks:
            pitched = np.zeros((0, pr.NUM_PITCHES, pr.T_PER_FRAGMENT), dtype=np.float32)
            track_ids = np.array([], dtype=np.int32)
        else:
            pitched = np.zeros(
                (len(pitched_tracks), pr.NUM_PITCHES, pr.T_PER_FRAGMENT),
                dtype=np.float32,
            )
            track_ids = np.array(pitched_tracks, dtype=np.int32)
            idx = {tid: i for i, tid in enumerate(pitched_tracks)}
            for n in notes:
                if n.is_drum:
                    continue
                i = idx[int(n.track_id)]
                start = pr.beats_to_steps(n.onset_beats)
                end = min(
                    pr.T_PER_FRAGMENT,
                    start + max(pr.beats_to_steps(n.duration_beats), 1),
                )
                if start >= pr.T_PER_FRAGMENT:
                    continue
                v = float(np.clip(n.velocity, 1, 127)) / 127.0
                pitched[i, n.pitch, start:end] = np.maximum(
                    pitched[i, n.pitch, start:end], v
                )
        drum = np.zeros((pr.NUM_PITCHES, pr.T_PER_FRAGMENT), dtype=np.float32)
        for n in notes:
            if not n.is_drum:
                continue
            start = pr.beats_to_steps(n.onset_beats)
            end = min(
                pr.T_PER_FRAGMENT, start + max(pr.beats_to_steps(n.duration_beats), 1)
            )
            if start >= pr.T_PER_FRAGMENT:
                continue
            v = float(np.clip(n.velocity, 1, 127)) / 127.0
            drum[n.pitch, start:end] = np.maximum(drum[n.pitch, start:end], v)
        out[str(item_id)] = {"pitched": pitched, "drum": drum, "track_ids": track_ids}
    return out


__all__ = [
    "SUBMISSION_COLUMNS",
    "PER_NOTE_COLUMNS",
    "SubmissionError",
    "notes_to_rows",
    "bundle_to_rows",
    "encode_notes",
    "decode_notes",
    "write_submission",
    "validate_submission",
    "rows_to_bundles",
]
