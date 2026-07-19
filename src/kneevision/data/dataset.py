from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image


class KneeXRayDataset(Dataset):
    def __init__(self, image_paths: list[Path], labels: list[int],
                 transform=None, minority_transform=None, minority_labels=None,
                 mixup_alpha: float = 0.0):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.minority_transform = minority_transform
        self.minority_labels = minority_labels or set()
        self.mixup_alpha = mixup_alpha

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.minority_transform and self.labels[idx] in self.minority_labels:
            img = self.minority_transform(img)
        elif self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class MixUpDataset(Dataset):
    """Wraps a base dataset and applies MixUp / CutMix on-the-fly."""
    def __init__(self, base_dataset: Dataset, alpha: float = 0.4, num_classes: int = 5):
        self.base = base_dataset
        self.alpha = alpha
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img1, label1 = self.base[idx]
        label1_onehot = torch.zeros(self.num_classes)
        label1_onehot[label1] = 1.0

        if self.alpha <= 0.0 or np.random.random() > 0.5:
            return img1, label1_onehot

        idx2 = np.random.randint(len(self.base))
        img2, label2 = self.base[idx2]
        label2_onehot = torch.zeros(self.num_classes)
        label2_onehot[label2] = 1.0

        lam = np.random.beta(self.alpha, self.alpha)

        if np.random.random() > 0.5:
            mixed_img = lam * img1 + (1 - lam) * img2
        else:
            cut_ratio = np.random.uniform(0.1, 0.3)
            h, w = img1.shape[1:]
            cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
            y = np.random.randint(0, h - cut_h + 1)
            x = np.random.randint(0, w - cut_w + 1)
            mixed_img = img1.clone()
            mixed_img[:, y:y + cut_h, x:x + cut_w] = img2[:, y:y + cut_h, x:x + cut_w] if isinstance(img2, torch.Tensor) else img2[:, y:y + cut_h, x:x + cut_w]

        mixed_label = lam * label1_onehot + (1 - lam) * label2_onehot
        return mixed_img, mixed_label


def make_weighted_sampler(labels: list[int], power: float = 0.5) -> WeightedRandomSampler:
    class_counts = torch.tensor([labels.count(i) for i in range(max(labels) + 1)], dtype=torch.float)
    weights = (1.0 / class_counts) ** power
    sample_weights = weights[torch.tensor(labels)]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)
