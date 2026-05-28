"""Conditional DDPM model for i-CLEVR."""

from __future__ import annotations

from typing import Any

from diffusers import UNet2DModel
from torch import Tensor, nn


class ConditionEmbedder(nn.Module):
    """Project multi-hot labels into UNet class-embedding space."""

    def __init__(self, num_classes: int = 24, condition_dim: int = 512) -> None:
        """Initialize label projector.

        Args:
            num_classes: Number of i-CLEVR object classes.
            condition_dim: Embedding dimension passed to UNet class conditioning.
        """
        super().__init__()
        self.proj = nn.Linear(num_classes, condition_dim)

    def forward(self, labels_multihot: Tensor) -> Tensor:
        """Convert labels from [B, 24] to [B, condition_dim]."""
        return self.proj(labels_multihot.float())


class ConditionalDDPM(nn.Module):
    """Conditional UNet2DModel following the referenced hw03 setup."""

    def __init__(
        self,
        image_size: int = 64,
        image_channels: int = 3,
        num_classes: int = 24,
        condition_dim: int = 512,
    ) -> None:
        """Initialize conditional DDPM backbone.

        Args:
            image_size: Input/output image resolution.
            image_channels: Number of image channels.
            num_classes: Number of condition classes.
            condition_dim: Class embedding dimension.
        """
        super().__init__()
        self.condition_embedder = ConditionEmbedder(
            num_classes=num_classes,
            condition_dim=condition_dim,
        )
        self.unet: Any = UNet2DModel(
            sample_size=image_size,
            in_channels=image_channels,
            out_channels=image_channels,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            class_embed_type="identity",
        )

    def forward(self, noisy_images: Tensor, timesteps: Tensor, labels_multihot: Tensor) -> Tensor:
        """Predict diffusion noise epsilon.

        Args:
            noisy_images: Noisy image tensor, shape [B, 3, H, W].
            timesteps: Diffusion timesteps tensor.
            labels_multihot: Multi-hot labels tensor, shape [B, 24].

        Returns:
            Predicted noise tensor with shape [B, 3, H, W].
        """
        class_embeds = self.condition_embedder(labels_multihot)
        return self.unet(noisy_images, timesteps, class_labels=class_embeds).sample
