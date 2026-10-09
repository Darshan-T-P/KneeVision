import pytest
import torch

from kneevision.training.losses import (
    FocalLoss,
    OrdinalLoss,
    ordinal_to_class,
    ordinal_to_probs,
)


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


def _one_hot(cls: int, k: int = 5) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor([cls]), k).float()


# ── Soft-target CORAL (opt-in) ────────────────────────────────────────────────


def test_ordinal_soft_onehot_equals_hard():
    """Test A: soft one-hot targets reproduce the hard path for every class."""
    torch.manual_seed(3)
    soft = OrdinalLoss(num_classes=5, soft_targets=True)
    hard = OrdinalLoss(num_classes=5, soft_targets=False)
    logits = torch.randn(1, 4)
    for cls in range(5):
        s = soft(logits, _one_hot(cls))
        h = hard(logits, torch.tensor([cls]))
        assert torch.allclose(s, h, atol=1e-6), f"one-hot identity failed for class {cls}"


def test_ordinal_soft_linearity_in_mix():
    """Test B: soft_CORAL(logits, lam*e_a + (1-lam)*e_b) ==
    lam*hard_CORAL(logits, a) + (1-lam)*hard_CORAL(logits, b).

    BCE is affine in the target, so the expected cumulative-tail target is the
    exact expected hard CORAL loss over the mix distribution.
    """
    torch.manual_seed(4)
    soft = OrdinalLoss(num_classes=5, soft_targets=True)
    hard = OrdinalLoss(num_classes=5, soft_targets=False)
    cases = [(1, 3, 0.7), (0, 4, 0.5), (2, 2, 0.3), (0, 1, 0.9), (3, 4, 0.4)]
    for a, b, lam in cases:
        for _ in range(5):
            logits = torch.randn(1, 4)
            p = lam * _one_hot(a) + (1 - lam) * _one_hot(b)
            s = soft(logits, p)
            h = lam * hard(logits, torch.tensor([a])) + (1 - lam) * hard(logits, torch.tensor([b]))
            assert torch.allclose(s, h, atol=1e-6), f"linearity failed for ({a},{b},{lam})"


def test_ordinal_soft_differs_from_argmax_hardening():
    """Test C: canary — on a genuinely mixed target the soft path must NOT equal
    the old argmax-hardened loss (the confound is real and removed)."""
    torch.manual_seed(5)
    soft = OrdinalLoss(num_classes=5, soft_targets=True)
    hard = OrdinalLoss(num_classes=5, soft_targets=False)  # argmax -> class 1
    logits = torch.randn(1, 4)
    p = 0.7 * _one_hot(1) + 0.3 * _one_hot(3)  # [0, 0.7, 0, 0.3, 0]
    s_val = soft(logits, p)
    h_val = hard(logits, p)
    assert not torch.allclose(s_val, h_val, atol=1e-6)
    # And the soft loss equals BCE against the explicit cumulative-tail target.
    expected_t = torch.tensor([[1.0, 0.3, 0.3, 0.0]])
    direct = torch.nn.functional.binary_cross_entropy_with_logits(logits, expected_t)
    assert torch.allclose(s_val, direct, atol=1e-6)


def test_ordinal_soft_valid_distribution_and_ordering():
    """Test D: rows sum to 1, cumulative targets in [0, 1], thresholds are
    non-increasing (P(Y > k) ordering preserved)."""
    torch.manual_seed(6)
    p = torch.tensor([[0.2, 0.3, 0.1, 0.25, 0.15]])
    assert torch.allclose(p.sum(dim=1), torch.ones(1), atol=1e-6)
    soft = OrdinalLoss(num_classes=5, soft_targets=True)
    logits = torch.randn(1, 4)
    loss = soft(logits, p)
    t = (1.0 - p.cumsum(dim=1))[:, :4]
    assert t.min() >= 0.0 - 1e-6 and t.max() <= 1.0 + 1e-6
    assert (t.diff(dim=1) <= 1e-6).all()  # non-increasing thresholds
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_ordinal_soft_invalid_width_raises():
    """A soft target whose columns != num_classes fails clearly instead of
    silently truncating."""
    soft = OrdinalLoss(num_classes=5, soft_targets=True)
    logits = torch.randn(2, 4)
    with pytest.raises(ValueError, match="5 columns"):
        soft(logits, torch.rand(2, 6))


def test_ordinal_default_still_argmax_hardening():
    """Default (soft_targets=False) keeps bit-compatible argmax behavior."""
    torch.manual_seed(7)
    criterion = OrdinalLoss(num_classes=5)
    logits = torch.randn(4, 4)
    p = torch.tensor([[0.0, 0.7, 0.0, 0.3, 0.0]] * 4)
    soft_p = criterion(logits, p)
    hard = criterion(logits, torch.full((4,), 1))  # argmax of p is class 1
    assert torch.allclose(soft_p, hard, atol=1e-6)
