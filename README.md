# AIA500500 Deep Learning - NYCU

## Course Information

- **Course:** AIA500500 Deep Learning
- **Language:** Python
- **Term:** 2026 Spring
- **Repository scope:** hw00-hw03 source code, provided assignment handouts, course slides, final exam PDFs, submitted reports, and selected generated results.

This repository collects the programming assignments for the Deep Learning course. The included work covers PyTorch warm-up material, convolutional neural networks for semantic segmentation, value-based reinforcement learning with DQN variants, and conditional diffusion models for image generation.

## Repository Layout

| Path | Topic | Main files |
| --- | --- | --- |
| `hw00/` | PyTorch warm-up and course preparation material. | provided warm-up PDFs |
| `hw01/` | Convolutional Nets: Oxford-IIIT Pet semantic segmentation with UNet and ResNet34UNet. | `src/train.py`, `src/inference.py`, `src/models/`, `docs/hw01_asignment.pdf` |
| `hw02/` | Value-Based Reinforcement Learning: DQN for CartPole and Pong, plus Double DQN, PER, and n-step returns. | `src/train_task1_cartpole.py`, `src/train_task2_pong_dqn.py`, `src/train_task3_pong_enhanced.py`, `src/eval_pong.py` |
| `hw03/` | Generative Models: conditional DDPM for i-CLEVR image generation. | `src/train.py`, `src/test.py`, `src/model.py`, `src/evaluator.py` |
| `slides/` | Course lecture slides. | `L01-Introduction.pdf` through `L17-Model-Based-Reinforcement-Learning.pdf` |
| `final exam/` | Final exam reference PDFs. | `DLP_2020_spring_final_exam.pdf` through `DLP_2025_spring_final_exam.pdf` |
| `Syllabus.pdf` | Course syllabus. | provided syllabus PDF |

## Requirements

- Python 3.12 or newer is recommended.
- A POSIX-like shell is useful for the example commands below.
- Each homework directory keeps its own `requirements.txt`.

Install packages for the homework you want to run:

```sh
cd hw01
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

```sh
cd hw02
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

```sh
cd hw03
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run Examples

For hw01 segmentation:

```sh
cd hw01
python -m src.train --model-name UNet --device cuda --num-epochs 250 --batch-size 16
python -m src.inference --model-name ResNet34UNet --timestamp 20260331-094415 --use-swa
```

For hw02 reinforcement learning:

```sh
cd hw02
python src/train_task1_cartpole.py
python src/eval_cartpole.py --model-path checkpoints/hw02_task1.pt
python src/train_task3_pong_enhanced.py
python src/eval_pong.py --model-path checkpoints/hw02_task3_best.pt
```

For hw03 conditional diffusion:

```sh
cd hw03
python src/train.py --batch-size 64 --epochs 100 --accum-grad 1 --wandb-mode online
python src/test.py --checkpoint-path checkpoints/<ddpm_checkpoint>.pth
```

## Data and Outputs

- `hw01/data/oxford-iiit-pet/` contains the Oxford-IIIT Pet dataset layout used by `hw01/src/`.
- `hw01/checkpoints/` stores segmentation checkpoints, training records, plots, configs, and generated submission CSV files.
- `hw02/checkpoints/` stores submitted DQN model snapshots.
- `hw02/outputs/results/` stores generated reinforcement-learning outputs and rendered videos.
- `hw03/data/` contains i-CLEVR images and metadata JSON files.
- `hw03/checkpoints/evaluator/` contains the evaluator checkpoint used by `hw03/src/evaluator.py`.
- `hw03/outputs/images/` stores generated images and image grids.
- `slides/` stores course lecture PDFs.
- `final exam/` stores final exam reference PDFs.

## File Naming Notes

- `hwxx/README.md` files summarize each homework directory.
- `src/` contains runnable source code.
- `docs/` contains assignment handouts, reports, and notes.
- `data/` contains local datasets or metadata files.
- `checkpoints/` contains model checkpoints and checkpoint-side records.
- `outputs/` contains generated images, videos, figures, or submissions.
- `logs/` contains local training logs.

## Course Scope Note

The course includes four homework assignments. This repository currently includes hw00 preparation material, organized homework directories through `hw03/`, course slides, and final exam reference PDFs.
