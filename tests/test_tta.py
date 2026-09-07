"""Tests for evaluation/tta.py — Test-Time Augmentation prediction."""
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from kneevision.evaluation.tta import tta_predict


# ── Minimal fake models ────────────────────────────────────────────────────────

class _FakeClassifier(nn.Module):
    """Always predicts grade 2 (softmax model)."""
    def __init__(self):
        super().__init__()
        self.ordinal = False

    def forward(self, x):
        batch = x.shape[0]
        logits = torch.zeros(batch, 5)
        logits[:, 2] = 10.0  # very high logit → grade 2
        return logits


class _FakeOrdinalClassifier(nn.Module):
    """Ordinal model that always predicts all thresholds exceeded → grade 4."""
    def __init__(self):
        super().__init__()
        self.ordinal = True

    def forward(self, x):
        batch = x.shape[0]
        return torch.ones(batch, 4) * 10.0  # all thresholds exceeded → grade 4


def _rand_pil(h=224, w=224):
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ── tta_predict (softmax model) ────────────────────────────────────────────────

def test_tta_predict_returns_tuple():
    model = _FakeClassifier()
    img = _rand_pil()
    result = tta_predict(model, img, device=torch.device("cpu"))
    assert isinstance(result, tuple) and len(result) == 2


def test_tta_predict_grade_range():
    model = _FakeClassifier()
    img = _rand_pil()
    pred, conf = tta_predict(model, img, device=torch.device("cpu"))
    assert 0 <= pred <= 4
    assert 0.0 <= conf <= 1.0


def test_tta_predict_correct_class_softmax():
    """Softmax model strongly predicting grade 2 should return pred=2."""
    model = _FakeClassifier()
    img = _rand_pil()
    pred, conf = tta_predict(model, img, device=torch.device("cpu"))
    assert pred == 2
    assert conf > 0.9


def test_tta_predict_confidence_is_probability():
    """Confidence must be in [0, 1]."""
    model = _FakeClassifier()
    for _ in range(5):
        _, conf = tta_predict(model, _rand_pil(), device=torch.device("cpu"))
        assert 0.0 <= conf <= 1.0


# ── tta_predict (ordinal model) ────────────────────────────────────────────────

def test_tta_predict_ordinal_grade_range():
    model = _FakeOrdinalClassifier()
    img = _rand_pil()
    pred, conf = tta_predict(model, img, device=torch.device("cpu"))
    assert 0 <= pred <= 4
    assert 0.0 <= conf <= 1.0


def test_tta_predict_ordinal_all_thresholds_exceeded():
    """All ordinal thresholds exceeded → grade 4."""
    model = _FakeOrdinalClassifier()
    img = _rand_pil()
    pred, conf = tta_predict(model, img, device=torch.device("cpu"))
    assert pred == 4


def test_tta_predict_ordinal_no_thresholds():
    """No ordinal thresholds exceeded → grade 0."""
    class _GradeZeroOrdinal(nn.Module):
        ordinal = True
        def forward(self, x):
            return torch.ones(x.shape[0], 4) * -10.0

    model = _GradeZeroOrdinal()
    pred, _ = tta_predict(model, _rand_pil(), device=torch.device("cpu"))
    assert pred == 0
