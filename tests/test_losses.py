import torch

from kneevision.training.losses import FocalLoss, OrdinalLoss, ordinal_to_class, ordinal_to_probs


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


def test_ordinal_to_probs_sums_to_one():
    logits = torch.randn(16, 4)
    probs = ordinal_to_probs(logits)
    assert probs.shape == (16, 5)
    assert torch.allclose(probs.sum(dim=1), torch.ones(16), atol=1e-5)
    assert (probs >= 0).all()


def test_ordinal_to_probs_never_negative_on_non_monotonic_logits():
    # Regression test: independently-trained binary heads aren't guaranteed
    # monotonic (e.g. P(grade>=2) > P(grade>=1)), which previously produced
    # negative "probabilities" once exposed as raw API JSON.
    logits = torch.tensor([[5.0, -5.0, 5.0, -5.0]])  # deliberately non-monotonic sigmoids
    probs = ordinal_to_probs(logits)
    assert (probs >= 0).all()
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_ordinal_to_probs_matches_class_extremes():
    # All thresholds strongly exceeded -> all mass on grade 4
    probs = ordinal_to_probs(torch.ones(1, 4) * 10)
    assert torch.allclose(probs, torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]]), atol=1e-3)
    # No threshold exceeded -> all mass on grade 0
    probs = ordinal_to_probs(torch.ones(1, 4) * -10)
    assert torch.allclose(probs, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]), atol=1e-3)
