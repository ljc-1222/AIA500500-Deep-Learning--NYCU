# hw02 - Value-Based Reinforcement Learning

## Overview

This assignment implements Deep Q-Network agents for CartPole and Atari Pong. Task 1 trains a vanilla DQN on `CartPole-v1`, Task 2 extends vanilla DQN to visual Pong observations, and Task 3 adds Double DQN, Prioritized Experience Replay, and 3-step returns.

## Assignment Scope

- Implement vanilla DQN with replay memory, epsilon-greedy exploration, and a target network.
- Preprocess Atari frames with grayscale conversion, 84x84 resize, and 4-frame stacking.
- Compare Task 3 enhancements against vanilla Pong DQN.
- Evaluate model snapshots over fixed seeds 0 through 19 and provide reproducible commands.

## Layout

| Path | Purpose |
| --- | --- |
| `src/` | Training, evaluation, starter DQN, and video-rendering scripts. |
| `checkpoints/` | Submitted Task 1, Task 2, Task 3 fixed-step, and best model snapshots. |
| `outputs/results/` | Training outputs produced by the scripts. |
| `logs/wandb/` | Existing W&B run logs. |
| `docs/` | Assignment spec (`hw02_assignment.pdf`), submitted report, slides, and demo video. |
| `requirements.txt` | Python package versions used by this homework. |

## Environment

```sh
cd hw02
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The existing local environment uses Python 3.12.3.

## Run

Task 1 training and evaluation:

```sh
python src/train_task1_cartpole.py
python src/eval_cartpole.py --model-path checkpoints/hw02_task1.pt
```

Task 2 training and evaluation:

```sh
python src/train_task2_pong_dqn.py
python src/eval_pong.py --model-path checkpoints/hw02_task2.pt
```

Task 3 training and fixed-snapshot evaluation:

```sh
python src/train_task3_pong_enhanced.py
python src/eval_pong.py --model-path checkpoints/hw02_task3_600000.pt
python src/eval_pong.py --model-path checkpoints/hw02_task3_1000000.pt
python src/eval_pong.py --model-path checkpoints/hw02_task3_1500000.pt
python src/eval_pong.py --model-path checkpoints/hw02_task3_2000000.pt
python src/eval_pong.py --model-path checkpoints/hw02_task3_2500000.pt
python src/eval_pong.py --model-path checkpoints/hw02_task3_best.pt
```

Ablation commands:

```sh
python src/train_task3_pong_enhanced.py --use-double-dqn --no-per --no-n-step
python src/train_task3_pong_enhanced.py --use-per --no-double-dqn --no-n-step
python src/train_task3_pong_enhanced.py --use-n-step --no-double-dqn --no-per
```

Render Pong evaluation videos:

```sh
python src/test_model.py --model-path checkpoints/hw02_task3_best.pt
```

## Current Artifacts

- Task 1 report result: 500.00 +/- 0.00 average reward over 20 fixed seeds.
- Task 2 report result: 19.40 +/- 1.50 average reward over 20 fixed seeds.
- Task 3 report result: best model 19.70 +/- 1.19; fixed checkpoints range from 600k to 2.5M environment steps.
- Source code is normalized to `src/` for consistency across homework folders.
