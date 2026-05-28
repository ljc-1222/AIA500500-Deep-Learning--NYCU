# hw01 - Convolutional Nets

## Overview

This assignment implements binary semantic segmentation on the Oxford-IIIT Pet dataset. The code trains and compares a plain UNet and a ResNet34-UNet, then exports run-length encoded submission CSV files from saved checkpoints.

## Assignment Scope

- Implement convolutional segmentation models for pet foreground masks.
- Train UNet and ResNet34UNet with fixed splits from `data/oxford-iiit-pet/`.
- Use BCE, Dice, and Lovasz losses with warmup, cosine scheduling, SWA, and early stopping.
- Generate submission CSV files from the trained checkpoints.

## Layout

| Path | Purpose |
| --- | --- |
| `src/` | Training, inference, dataset, loss, utility, and model source code. |
| `data/oxford-iiit-pet/` | Oxford-IIIT Pet images, annotations, and split files. |
| `checkpoints/` | Saved UNet and ResNet34UNet checkpoints, training CSVs, plots, configs, and submissions. |
| `docs/` | Submitted short technical summary. |
| `requirements.txt` | Python package versions used by this homework. |

## Environment

```sh
cd hw01
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The existing local environment uses Python 3.13.5. CUDA is recommended for training; the saved training configs were produced on CUDA with bfloat16 autocast enabled.

## Run

Train UNet:

```sh
python -m src.train \
  --model-name UNet \
  --device cuda \
  --num-epochs 250 \
  --batch-size 16
```

Train ResNet34UNet:

```sh
python -m src.train \
  --model-name ResNet34UNet \
  --device cuda \
  --num-epochs 250 \
  --batch-size 16 \
  --learning-rate 1.5e-4 \
  --swa-lr 1e-5
```

Generate submissions from existing checkpoints:

```sh
python -m src.inference --model-name UNet --timestamp 20260331-100231 --use-swa
python -m src.inference --model-name ResNet34UNet --timestamp 20260331-094415 --use-swa
```

## Current Artifacts

- Split sizes: 5173 train, 739 validation, and 739 test entries for each submitted model split.
- UNet best validation soft Dice in the saved CSV: 0.892200 at epoch 137.
- ResNet34UNet best validation soft Dice in the saved CSV: 0.912567 at epoch 142.
- Existing submissions are stored beside their checkpoints under `checkpoints/<model>/<timestamp>/`.
