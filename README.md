# Hackathon — one-shot music style transfer

PyTorch reimplementation of Groove2Groove adapted for the Kaggle hackathon
on one-shot symbolic music style transfer.

## Layout

```
hackaton/
  g2g_pytorch/         PyTorch model + train / infer entry points
    config.py          all hyperparameters (single dataclass)
    dataset.py         NPZ -> tensor; manifest parsing
    metrics.py         in-loop CP / SF / harmonic-mean scoring
    models/            ContentEncoder + StyleEncoder + RollDecoder
    train.py           training entry point
    infer.py           submission CSV writer
    train_colab.ipynb  ready-to-run Colab notebook
    README.md          architecture + module-level docs
  utility/             challenge-supplied helpers (do not edit)
    pianoroll.py       roll <-> notes <-> midi
    metric.py          CP, SF, harmonic-mean scorer
    submission.py      Kaggle wire-format writer + validator
    data.py            manifest + bundle helpers
  requirements.txt
```

## Quick start (local, GPU machine)

```bash
pip install -r requirements.txt

# put manifest.csv, rolls/ and style_profiles/ under ./dataset/
python -m g2g_pytorch.train --dataset-root ./dataset --logdir ./runs/exp1
python -m g2g_pytorch.infer --dataset-root ./dataset \
    --ckpt ./runs/exp1/model.pt --output submission.csv
```

A one-minute smoke test:

```bash
python -m g2g_pytorch.train --debug --dataset-root ./dataset --logdir ./runs/smoke
```

## Quick start (Colab)

Open `g2g_pytorch/train_colab.ipynb`, edit the four paths in cell 0
(GitHub URL, `DRIVE_DATASET_ZIP`, `DRIVE_RUNS_DIR`, `RUN_NAME`), then run
cells top-to-bottom. The notebook handles Drive → local SSD copy, train,
infer and writing the submission back to Drive.

See [g2g_pytorch/README.md](g2g_pytorch/README.md) for the architecture
and the known divergences from the original Groove2Groove paper.
