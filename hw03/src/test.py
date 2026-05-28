"""Generate test images and evaluate accuracy with TA evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DDPMScheduler
from torchvision.utils import make_grid, save_image

from dataloader import TestConditionDataset
from evaluator import evaluation_model
from model import ConditionalDDPM
from utils import (
    build_noise_scheduler,
    denormalize,
    load_ddpm_checkpoint,
    load_objects_map,
    sample_images,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "images"


def parse_args() -> argparse.Namespace:
    """Parse evaluation options."""
    parser = argparse.ArgumentParser(description="Generate i-CLEVR test images and evaluate them.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate_one_split(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    split_file: str,
    batch_size: int,
    image_size: int,
    device: torch.device,
    out_dir: Path,
    evaluator: evaluation_model,
) -> float:
    """Generate images for one split and compute evaluator accuracy."""
    dataset = TestConditionDataset(condition_file=split_file)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    split_name = split_file.replace(".json", "")
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    all_images: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    image_index = 0
    with torch.no_grad():
        for labels in dataloader:
            labels = labels.to(device)
            samples, _ = sample_images(
                model=model,
                scheduler=scheduler,
                conditions=labels,
                image_size=image_size,
                device=device,
            )
            all_images.append(samples.detach().cpu())
            all_labels.append(labels.detach().cpu())
            for image in denormalize(samples).cpu():
                save_image(image, split_dir / f"{image_index:03d}.png")
                image_index += 1

    merged_images = torch.cat(all_images, dim=0)
    merged_labels = torch.cat(all_labels, dim=0)
    accuracy = float(evaluator.eval(merged_images.to(device), merged_labels.to(device)))

    grid = make_grid(denormalize(merged_images), nrow=8)
    save_image(grid, split_dir / f"{split_name}_grid.png")
    return accuracy


def save_denoising_process(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    root: Path,
    image_size: int,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Save denoising-process grid for the report scene ["red sphere", "cyan cylinder", "cyan cube"]."""
    _, denoise_steps = sample_images(
        model=model,
        scheduler=scheduler,
        image_size=image_size,
        device=device,
        save_denoise_steps=True,
        label_names=["red sphere", "cyan cylinder", "cyan cube"],
        objects_map=load_objects_map(root),
    )
    grid = make_grid(denormalize(torch.cat(denoise_steps, dim=0)), nrow=len(denoise_steps))
    save_image(grid, out_dir / "denoise_process.png")


def main() -> None:
    """Entry point for full evaluation on both test splits."""
    args = parse_args()
    # set_seed(1739178872)
    # set_seed(107420369)
    # set_seed(716893640)
    # set_seed(286369870)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalDDPM()
    load_ddpm_checkpoint(model, str(args.checkpoint_path), device)
    scheduler = build_noise_scheduler(num_inference_steps=args.num_inference_steps)
    evaluator = evaluation_model(device=device)

    print("Evaluating test.json...")
    test_acc = evaluate_one_split(
        model=model,
        scheduler=scheduler,
        split_file="test.json",
        batch_size=args.batch_size,
        image_size=64,
        device=device,
        out_dir=args.output_dir,
        evaluator=evaluator,
    )
    
    print("Evaluating new_test.json...")
    new_test_acc = evaluate_one_split(
        model=model,
        scheduler=scheduler,
        split_file="new_test.json",
        batch_size=args.batch_size,
        image_size=64,
        device=device,
        out_dir=args.output_dir,
        evaluator=evaluator,
    )
    print("Saving denoising process...")
    save_denoising_process(
        model=model,
        scheduler=scheduler,
        root=PROJECT_ROOT,
        image_size=64,
        device=device,
        out_dir=args.output_dir,
    )

    print(f"test.json accuracy: {test_acc:.4f}")
    print(f"new_test.json accuracy: {new_test_acc:.4f}")


if __name__ == "__main__":
    main()
