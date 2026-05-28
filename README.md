# AIA500500 Deep Learning - NYCU

## Course Information

- Course: Deep Learning
- Term: Spring 2026
- School: National Yang Ming Chiao Tung University
- Language: English
- Evaluation from syllabus: 4 homework assignments, 80%; final exam, 20%.
- Hardware requirement from syllabus: GPU access with at least 6 GB memory.

This repository collects local course materials, assignment source code, reports, trained artifacts, and generated outputs for the Deep Learning course. The organized homework directories use a shared structure: `src/`, `docs/`, `data/`, `checkpoints/`, `outputs/`, and `logs/` when applicable.

## Repository Layout

| Path | Topic | Main contents |
| --- | --- | --- |
| `hw00/` | PyTorch warm-up material | Provided hw00 PDFs. |
| `hw01/` | hw01, Convolutional Nets | Oxford-IIIT Pet semantic segmentation with UNet and ResNet34UNet. |
| `hw02/` | hw02, Value-Based Reinforcement Learning | DQN for CartPole and Pong, plus Double DQN, PER, and multi-step returns. |
| `hw03/` | hw03, Generative Models | Conditional DDPM for i-CLEVR image generation. |
| `slides/` | Lecture slides | Course lecture PDFs from introduction through reinforcement learning. |
| `Syllabus.pdf` | Course syllabus | Course schedule, grading policy, and requirements. |

## Homework Layout Convention

Each organized homework uses these paths as consistently as the original artifacts allow:

| Path | Meaning |
| --- | --- |
| `README.md` | Assignment-specific summary, layout, environment, commands, and current artifacts. |
| `src/` | Source code and runnable scripts. |
| `docs/` | Assignment handouts, reports, slides, videos, and submission archives. |
| `data/` | Local datasets and metadata files. |
| `checkpoints/` | Model checkpoints and checkpoint-side training records. |
| `outputs/` | Generated figures, videos, submissions, and sample outputs. |
| `logs/` | Training logs such as W&B run directories. |
| `requirements.txt` | Assignment-specific Python dependencies, when available. |

## Assignments

| Homework | Assignment | Summary | README |
| --- | --- | --- | --- |
| `hw01/` | hw01: Convolutional Nets | Binary pet-mask segmentation with UNet and ResNet34UNet. | `hw01/README.md` |
| `hw02/` | hw02: Value-Based RL | DQN on CartPole and Pong; enhanced Pong DQN with Double DQN, PER, and n-step returns. | `hw02/README.md` |
| `hw03/` | hw03: Generative Models | Conditional DDPM on i-CLEVR with evaluator-based accuracy. | `hw03/README.md` |

## Environment

Use the assignment-specific environment file when running code:

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

The local `.venv/` folders are kept in each homework directory for convenience but are ignored by Git.

## Artifact Policy

Large datasets, trained checkpoints, generated outputs, W&B logs, videos, and submission archives are organized in the relevant homework folders. They are intentionally excluded by `.gitignore` patterns so source code and documentation can remain reviewable without committing heavy experiment artifacts.
