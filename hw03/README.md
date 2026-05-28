# hw03 - Generative Models

## Overview

This assignment implements a conditional Denoising Diffusion Probabilistic Model for i-CLEVR. The model receives multi-label object conditions and generates 64x64 RGB images that are evaluated by the TA-provided pretrained ResNet18 classifier.

## Assignment Scope

- Implement an i-CLEVR dataloader and multi-hot condition representation.
- Train a conditional DDPM with a Diffusers `UNet2DModel` backbone.
- Use a squared-cosine DDPM noise schedule and epsilon prediction.
- Generate images for `test.json` and `new_test.json`, save 8x4 grids, and save a denoising-process grid for `["red sphere", "cyan cylinder", "cyan cube"]`.
- Evaluate generated images with the provided classifier checkpoint.

## Layout

| Path | Purpose |
| --- | --- |
| `src/` | Dataset, model, training, sampling, evaluation, and utility source code. |
| `data/` | i-CLEVR training images plus `train.json`, `test.json`, `new_test.json`, and `objects.json`. |
| `checkpoints/evaluator/` | TA-provided classifier checkpoint. |
| `checkpoints/` | Target location for trained DDPM checkpoints. |
| `outputs/images/` | Generated images, image grids, and denoising-process visualizations. |
| `logs/wandb/` | Existing W&B run logs. |
| `docs/` | Assignment spec (`hw03_assignment.pdf`), assignment slides (`hw03_assignment_slides.pptx`), submitted report, original submission archive, and dataset notes. |
| `requirements.txt` | Python package versions used by this homework. |

## Environment

```sh
cd hw03
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The existing local environment uses Python 3.12.3.

## Run

Train the DDPM:

```sh
python src/train.py \
  --batch-size 64 \
  --epochs 100 \
  --accum-grad 1 \
  --wandb-mode online
```

For offline local runs:

```sh
python src/train.py --wandb-mode offline
```

Generate and evaluate images:

```sh
python src/test.py --checkpoint-path checkpoints/<ddpm_checkpoint>.pth
```

`src/test.py` always uses `checkpoints/evaluator/checkpoint.pth` as the TA evaluator checkpoint. The DDPM generator checkpoint is not included in this working tree, so it must be provided with `--checkpoint-path`.

## Current Artifacts

- Dataset metadata: 18009 training entries, 32 `test.json` conditions, 32 `new_test.json` conditions, and 24 object labels.
- Reported final DDPM checkpoint: epoch 98.
- Evaluator checkpoint: `checkpoints/evaluator/checkpoint.pth`.
- Reported evaluator accuracies: 0.8066 on `test.json` and 0.8333 on `new_test.json`.
- Existing generated images and grids are stored under `outputs/images/`.
