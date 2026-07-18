from pathlib import Path
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image


class KneeXRayDataset(Dataset):
    def __init__(self, image_paths: list[Path], labels: list[int],
                 transform=None, minority_transform=None, minority_labels=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.minority_transform = minority_transform
        self.minority_labels = minority_labels or set()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.minority_transform and self.labels[idx] in self.minority_labels:
            img = self.minority_transform(img)
        elif self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def make_weighted_sampler(labels: list[int], power: float = 0.3) -> WeightedRandomSampler:
    class_counts = torch.tensor([labels.count(i) for i in range(max(labels) + 1)], dtype=torch.float)
    weights = (1.0 / class_counts) ** power
    sample_weights = weights[torch.tensor(labels)]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)
