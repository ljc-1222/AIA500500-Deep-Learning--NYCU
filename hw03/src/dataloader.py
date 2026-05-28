"""Dataset definitions for the i-CLEVR conditional generation task."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from utils import labels_to_multihot, load_objects_map

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_image_transform() -> transforms.Compose:
    """Create the default image transform required by the evaluator."""
    
    return transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class iCLEVRDataset(Dataset[tuple[Tensor, Tensor]]):
    """i-CLEVR training dataset for conditional DDPM.

    This dataset returns paired training samples:
    - image tensor: normalized to [-1, 1] with shape [3, 64, 64]
    - condition tensor: multi-hot vector with shape [24]
    """

    def __init__(
        self,
        transform: transforms.Compose | None = None,
    ) -> None:
        """Initialize training dataset.

        Args:
            transform: Image transform pipeline. Defaults to evaluator-compatible
                transform when None.
        """
        self.root = PROJECT_ROOT
        self.transform = transform or _default_image_transform()
        self.objects_map = load_objects_map(self.root)
        self.num_classes = len(self.objects_map)
        self.images_dir = self.root / "data"

        with (self.root / "data" / "train.json").open("r", encoding="utf-8") as file:
            raw_data: dict[str, list[str]] = json.load(file)
        self.samples: list[tuple[str, list[str]]] = list(raw_data.items())

    def __len__(self) -> int:
        """Return the number of training examples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Get one training sample.

        Args:
            index: Sample index.

        Returns:
            A tuple of image tensor and condition multi-hot tensor.
        """
        filename, labels = self.samples[index]
        image_path = self.images_dir / filename

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        condition = labels_to_multihot(labels, self.objects_map, self.num_classes)
        return image_tensor, condition


class TestConditionDataset(Dataset[Tensor]):
    """Condition-only dataset for test.json and new_test.json."""

    def __init__(self, condition_file: str = "test.json") -> None:
        """Initialize testing condition dataset.

        Args:
            condition_file: Name of condition file. Usually `test.json` or
                `new_test.json`.
        """
        self.root = PROJECT_ROOT
        self.objects_map = load_objects_map(self.root)
        self.num_classes = len(self.objects_map)

        with (self.root / "data" / condition_file).open("r", encoding="utf-8") as file:
            self.conditions: list[list[str]] = json.load(file)

    def __len__(self) -> int:
        """Return number of condition entries."""
        return len(self.conditions)

    def __getitem__(self, index: int) -> Tensor:
        """Get one condition entry as a multi-hot tensor."""
        labels = self.conditions[index]
        return labels_to_multihot(labels, self.objects_map, self.num_classes)


if __name__ == "__main__":
    
    train_dataset = iCLEVRDataset()
    train_image, train_condition = train_dataset[0]
    print("Train sample image shape:", train_image.shape)
    print("Train sample condition shape:", train_condition.shape)

    test_dataset = TestConditionDataset(condition_file="test.json")
    test_condition = test_dataset[0]
    print("Test sample condition shape:", test_condition.shape)
