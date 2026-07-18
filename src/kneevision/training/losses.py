import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return ( (1 - pt) ** self.gamma * ce ).mean()


class OrdinalLoss(nn.Module):
    """CORAL (Consistent Rank Logits) loss for ordinal regression.
    Treats KL grades as ordered: 0 < 1 < 2 < 3 < 4.
    Predicts 4 binary tasks: is grade > 0? > 1? > 2? > 3?"""
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, num_classes-1) — one binary logit per ordinal threshold
        # targets: (B,) — integer grades 0..4
        B = logits.shape[0]
        targets = targets.long()
        # Convert target grade to binary labels for each threshold
        # e.g. grade 3 -> [1, 1, 1, 0] (is >0? yes, >1? yes, >2? yes, >3? no)
        labels = torch.arange(self.num_classes - 1, device=targets.device).float()
        extended_targets = targets.float().unsqueeze(1)
        ordinal_labels = (extended_targets > labels).float()
        return F.binary_cross_entropy_with_logits(logits, ordinal_labels)


def ordinal_to_class(logits: torch.Tensor) -> torch.Tensor:
    """Convert ordinal logits to class predictions.
    Sum of binary predictions = predicted grade."""
    probs = torch.sigmoid(logits)
    return probs.round().sum(dim=1).long()
