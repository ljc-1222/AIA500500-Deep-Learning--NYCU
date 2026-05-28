"""Training script for i-CLEVR conditional DDPM baseline.

This script trains a conditional diffusion model with noise-prediction objective.
It also supports optional evaluator-guidance loss using TA-provided classifier.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from diffusers import DDPMScheduler
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import iCLEVRDataset
from evaluator import evaluation_model
from model import ConditionalDDPM
from utils import (
    build_noise_scheduler,
    reconstruct_x0,
    save_checkpoint,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    """Configuration for DDPM training."""

    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_train_timesteps: int = 1000
    mixed_precision: bool = False
    seed: int = 42
    save_every: int = 1
    evaluator_guidance_weight: float = 0.1
    max_grad_norm: float = 1.0
    accum_grad: int = 2
    wandb_project: str = "hw03-conditional-ddpm"
    wandb_run_name: str = ""
    wandb_entity: str = ""
    wandb_mode: str = "online"
    log_interval: int = 20
    onecycle_initial_lr: float = 1e-5
    onecycle_max_lr: float = 1e-4
    onecycle_min_lr: float = 5e-6
    onecycle_pct_start: float = 0.3


def parse_args() -> TrainConfig:
    """Parse command-line arguments into training config."""
    parser = argparse.ArgumentParser(description="Train conditional DDPM on i-CLEVR.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--onecycle-initial-lr", type=float, default=1e-5)
    parser.add_argument("--onecycle-max-lr", type=float, default=1e-4)
    parser.add_argument("--onecycle-min-lr", type=float, default=5e-6)
    parser.add_argument("--onecycle-pct-start", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default = False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--evaluator-guidance-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--accum-grad", type=int, default=1)
    parser.add_argument("--wandb-project", type=str, default="hw03-conditional-ddpm")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--log-interval", type=int, default=20)
    args = parser.parse_args()

    return TrainConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        onecycle_initial_lr=args.onecycle_initial_lr,
        onecycle_max_lr=args.onecycle_max_lr,
        onecycle_min_lr=args.onecycle_min_lr,
        onecycle_pct_start=args.onecycle_pct_start,
        weight_decay=args.weight_decay,
        num_train_timesteps=args.num_train_timesteps,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        save_every=args.save_every,
        evaluator_guidance_weight=args.evaluator_guidance_weight,
        max_grad_norm=args.max_grad_norm,
        accum_grad=max(1, args.accum_grad),
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        log_interval=args.log_interval,
    )


def count_optimizer_steps_per_epoch(num_batches: int, accum_grad: int) -> int:
    """Count optimizer updates in one epoch (including a tail step when batches do not divide accum)."""
    accum_grad = max(1, accum_grad)
    steps = num_batches // accum_grad
    if num_batches % accum_grad != 0:
        steps += 1
    return steps

def train_one_epoch(
    model: ConditionalDDPM,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: AdamW,
    lr_scheduler: OneCycleLR,
    scaler: torch.amp.GradScaler,
    scheduler: DDPMScheduler,
    evaluator_net: torch.nn.Module | None,
    config: TrainConfig,
    device: torch.device,
    epoch: int,
    global_step: int,
) -> int:
    """Train one epoch and return updated global step."""
    model.train()
    progress = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    running_noise_loss = 0.0
    running_guidance_loss = 0.0
    epoch_steps = 0

    optimizer.zero_grad(set_to_none=True)
    for batch_idx, (images, labels) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        use_amp = config.mixed_precision and device.type == "cuda"
        
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            predicted_noise = model(noisy_images, timesteps, labels)
            noise_loss = F.mse_loss(predicted_noise, noise)
            total_loss = noise_loss

            guidance_loss = torch.zeros((), device=device)
            if evaluator_net is not None and config.evaluator_guidance_weight > 0.0:
                x0_hat = reconstruct_x0(noisy_images, predicted_noise, timesteps, scheduler)
                evaluator_pred = evaluator_net(x0_hat)
                guidance_loss = F.binary_cross_entropy(evaluator_pred, labels)
                total_loss = total_loss + config.evaluator_guidance_weight * guidance_loss

        scaled_loss = total_loss / config.accum_grad
        scaler.scale(scaled_loss).backward()
        if batch_idx % config.accum_grad == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale():
                lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1
        epoch_steps += 1
        running_noise_loss += float(noise_loss.detach().item())
        running_guidance_loss += float(guidance_loss.detach().item())
        mean_noise_loss = running_noise_loss / epoch_steps
        mean_guidance_loss = running_guidance_loss / epoch_steps
        progress.set_postfix(
            noise_loss=f"{noise_loss.item():.4f}",
            guidance_loss=f"{guidance_loss.item():.4f}",
            avg_noise=f"{mean_noise_loss:.4f}",
            avg_guidance=f"{mean_guidance_loss:.4f}",
        )
        
        if global_step % config.log_interval == 0:
            wandb.log(
                {
                    "train/noise_loss_step": float(noise_loss.item()),
                    "train/guidance_loss_step": float(guidance_loss.item()),
                    "train/total_loss_step": float(total_loss.item()),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    "train/epoch": float(epoch),
                },
                step=global_step,
            )

    wandb.log(
        {
            "train/noise_loss_epoch": running_noise_loss / max(epoch_steps, 1),
            "train/guidance_loss_epoch": running_guidance_loss / max(epoch_steps, 1),
            "train/epoch_done": float(epoch),
        },
        step=global_step,
    )

    return global_step


def main() -> None:
    """Entry point for training."""
    config = parse_args()
    set_seed(config.seed)
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or None,
        entity=config.wandb_entity or None,
        mode=config.wandb_mode,
        config=vars(config),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = iCLEVRDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    div_factor = config.onecycle_max_lr / config.onecycle_initial_lr
    final_div_factor = config.onecycle_initial_lr / config.onecycle_min_lr

    model = ConditionalDDPM().to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.onecycle_max_lr / div_factor,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(enabled=config.mixed_precision and device.type == "cuda")
    noise_scheduler = build_noise_scheduler(config.num_train_timesteps)

    batches_per_epoch = len(dataloader)
    steps_per_epoch = count_optimizer_steps_per_epoch(batches_per_epoch, config.accum_grad)
    total_optimizer_steps = max(1, steps_per_epoch * config.epochs)

    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=config.onecycle_max_lr,
        total_steps=total_optimizer_steps,
        pct_start=config.onecycle_pct_start,
        anneal_strategy="cos",
        cycle_momentum=False,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )

    evaluator_net = evaluation_model(device=device)

    out_dir = PROJECT_ROOT / "checkpoints"
    global_step = 0
    
    for epoch in tqdm(range(config.epochs), desc="Training"):
        global_step = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            scheduler=noise_scheduler,
            evaluator_net=evaluator_net,
            config=config,
            device=device,
            epoch=epoch,
            global_step=global_step,
        )

        if (epoch + 1) % config.save_every == 0:
            ckpt_path = out_dir / f"ddpm_epoch_{epoch + 1:03d}.pth"
            save_checkpoint(
                path=ckpt_path,
                model=model,
                epoch=epoch,
                global_step=global_step,
            )
            wandb.log({"checkpoint/epoch": float(epoch + 1), "checkpoint/path": str(ckpt_path)})

    final_path = out_dir / "ddpm_last.pth"
    save_checkpoint(
        path=final_path,
        model=model,
        epoch=config.epochs - 1,
        global_step=global_step,
    )
    wandb.log({"checkpoint/final_path": str(final_path)})
    wandb.finish()


if __name__ == "__main__":
    main()
