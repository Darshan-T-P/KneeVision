import torch

from kneevision.training.losses import FocalLoss, OrdinalLoss, ordinal_to_class


def test_focal_loss_standard():
    criterion = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    loss = criterion(logits, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_focal_loss_soft_targets():
    criterion = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 5)
    targets = torch.nn.functional.one_hot(torch.randint(0, 5, (8,)), 5).float()
    loss = criterion(logits, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_ordinal_loss_shapes():
    criterion = OrdinalLoss(num_classes=5)
    logits = torch.randn(8, 4)
    targets = torch.randint(0, 5, (8,))
    loss = criterion(logits, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_ordinal_to_class_range():
    logits = torch.randn(64, 4)
    preds = ordinal_to_class(logits)
    assert preds.min() >= 0 and preds.max() <= 4


def test_ordinal_to_class_exact():
    # All thresholds exceeded -> grade 4
    logits = torch.ones(1, 4) * 10
    assert ordinal_to_class(logits).item() == 4
    # No threshold exceeded -> grade 0
    logits = torch.ones(1, 4) * -10
    assert ordinal_to_class(logits).item() == 0
