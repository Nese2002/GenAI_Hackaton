# Groove2Groove — PyTorch port for the hackathon

A roll-to-roll PyTorch reimplementation of the Groove2Groove style-transfer
model, adapted for the piano-roll NPZ format used by the Kaggle hackathon.

## Layout

```
g2g_pytorch/
  config.py            # all hyperparameters
  dataset.py           # NPZ -> tensor; manifest parsing; carve_val_from_train
  metrics.py           # wraps utility.metric for in-loop scoring
  models/
    encoders.py        # ContentEncoder (CNN+BiGRU), StyleEncoder (CNN+GRU)
    decoder.py         # RollDecoder (cross-attention + FiLM, non-AR)
    g2g.py             # end-to-end model
  train.py             # training entry point
  infer.py             # writes a Kaggle-format submission CSV
```

## Architecture

* **Content encoder** — 2-D CNN over the 2-channel piano roll
  ``(pitched_max, drum) × 128 × 128``, followed by a BiGRU on the flattened
  per-time-step features. Produces an encoder memory of shape
  ``(B, T', 2·H_content)``.
* **Style encoder** — same 2-D backbone on the Z roll, then a 1-D conv stack
  (300 filters, kernels 6/4/4, three 2× pools) and a GRU whose final state
  is projected to ``style_dim`` (256). Optional dropout.
* **Decoder** — non-autoregressive: 128 learned time-step queries + sinusoidal
  positions cross-attend to the content memory through ``decoder_layers``
  Transformer-like blocks, each with FiLM conditioning on the style vector.
  Two linear heads emit per-cell logits for ``Y_pitched`` and ``Y_drum``.

Loss: weighted ``BCEWithLogits`` on soft targets (the velocity itself in
[0,1]). ``pos_weight`` upweights active cells because piano rolls are
sparse.

## Training

```bash
cd code
python -m g2g_pytorch.train \
    --dataset-root ../dataset \
    --logdir runs/exp1 \
    --batch-size 16 \
    --max-steps 100000
```

If the local ``manifest.csv`` references val/test NPZs that are not on
disk, the trainer falls back to carving a held-out slice of the present
train items as a development val. The in-loop scorer then uses per-item
Y bundles to build a fallback style profile (only train styles lack a
precomputed profile under ``dataset/style_profiles/``).

For a quick sanity check:

```bash
python -m g2g_pytorch.train --debug --dataset-root ../dataset --logdir runs/smoke
```

(50 steps, tiny model, CPU-friendly — confirms loss decreases and val
scoring runs.)

## Inference + submission

```bash
cd code
python -m g2g_pytorch.infer \
    --dataset-root ../dataset \
    --ckpt runs/exp1/model.pt \
    --output submission.csv
```

The script runs the model on the ``test`` split, applies sigmoid +
threshold, converts each predicted roll to note records via
``utility.pianoroll.roll_to_notes``, and writes the Kaggle wire format
(``ID, notes``) using ``utility.submission.write_submission``. It then
round-trip-validates the CSV against the required item IDs.

## Known divergences from the original Groove2Groove paper

1. **Output representation** — the original model is an autoregressive
   token decoder over a custom BeatRelativeEncoding. This port directly
   predicts the velocity roll (a 128×128 dense output per channel) since
   the challenge scoring is roll-based.
2. **Style encoder input** — the paper embeds the Z note tokens; this
   port reads the Z piano roll directly through the same 2-D backbone as
   the content encoder. Simpler and avoids re-implementing the token
   codec.
3. **Per-track filters** — the original model emits one note-sequence
   per program filter (Bass / Piano / Guitar / Strings) and merges them.
   This port emits a single pitched roll + a single drum roll; the
   metric and the submission format do not require track separation.
4. **No tempo warping / chopping** — input bundles are already 8 bars
   on a fixed 16th-note grid.

## Dependencies

```
torch >= 2.0
numpy
pandas       # required by utility.submission
pretty_midi  # only needed if you also use utility.pianoroll.midi_to_notes
```
