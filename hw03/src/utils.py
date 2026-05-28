"""Shared utilities for i-CLEVR conditional DDPM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from diffusers import DDPMScheduler
from torch import Tensor, nn
from tqdm import tqdm


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_objects_map(root: str | Path) -> dict[str, int]:
    """Load the i-CLEVR object-to-index mapping.

    Args:
        root: Project root containing `data/objects.json`.

    Returns:
        Mapping from object label string to class index.
    """
    root = Path(root)
    objects_path = root / "data" / "objects.json"
    if not objects_path.exists():
        objects_path = root / "objects.json"
    with objects_path.open("r", encoding="utf-8") as file:
        objects_map: dict[str, int] = json.load(file)
    return objects_map


def labels_to_multihot(
    labels: Sequence[str],
    objects_map: dict[str, int],
    num_classes: int,
) -> Tensor:
    """Convert a list of object labels into a multi-hot tensor."""
    multihot = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        multihot[objects_map[label]] = 1.0
    return multihot


def denormalize(images: Tensor) -> Tensor:
    """Convert tensors normalized from [-1, 1] back to [0, 1]."""
    return images.mul(0.5).add(0.5).clamp(0.0, 1.0)


def build_noise_scheduler(
    num_train_timesteps: int = 1000,
    num_inference_steps: int | None = None,
) -> DDPMScheduler:
    """Build the shared DDPM scheduler used by training and inference."""
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    if num_inference_steps is not None:
        scheduler.set_timesteps(num_inference_steps)
    return scheduler


def reconstruct_x0(
    noisy_images: Tensor,
    predicted_noise: Tensor,
    timesteps: Tensor,
    scheduler: DDPMScheduler,
) -> Tensor:
    """Reconstruct x0 estimate from xt and predicted noise."""
    alpha_bar = scheduler.alphas_cumprod.to(noisy_images.device)[timesteps]
    sqrt_alpha_bar = alpha_bar.sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1)
    x0_hat = (noisy_images - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar
    return x0_hat.clamp(-1.0, 1.0)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    global_step: int,
) -> None:
    """Save model weights and lightweight training metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
        },
        path,
    )


def load_ddpm_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> nn.Module:
    """Load model weights from either training or inference checkpoint format."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_images(
    model: nn.Module,
    scheduler: DDPMScheduler,
    image_size: int,
    device: torch.device,
    conditions: Tensor | None = None,
    save_denoise_steps: bool = False,
    label_names: Sequence[str] | None = None,
    objects_map: dict[str, int] | None = None,
) -> tuple[Tensor, list[Tensor]]:
    """Generate images with reverse diffusion.

    Either pass a multi-hot ``conditions`` tensor (batched), or pass
    ``label_names`` + ``objects_map`` to build the conditioning for a
    single image inside the function.
    """

    if label_names is not None:
        conditions = (
            labels_to_multihot(label_names, objects_map, len(objects_map))
            .unsqueeze(0)
            .to(device)
        )
    batch_size = conditions.size(0)
    samples = torch.randn(batch_size, 3, image_size, image_size, device=device)
    denoise_steps: list[Tensor] = []
    total_steps = len(scheduler.timesteps)

    steps = total_steps // 7
    save_marks = {0, steps, 2 * steps, 3 * steps, 4 * steps, 5 * steps, 6 * steps, total_steps - 1}

    for step_idx, timestep in tqdm(enumerate(scheduler.timesteps), desc="Sampling"):
        timestep_batch = torch.full((batch_size,), int(timestep), device=device, dtype=torch.long)
        noise_pred = model(samples, timestep_batch, conditions)
        samples = scheduler.step(noise_pred, timestep, samples).prev_sample
        if save_denoise_steps and step_idx in save_marks:
            denoise_steps.append(samples.detach().cpu())

    return samples, denoise_steps
