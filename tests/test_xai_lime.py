"""Tests for xai/lime.py — segment grid, LIME explain output contract."""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from kneevision.xai.lime import _segment_grid, explain as lime_explain
from kneevision.data.transforms import val_transform


# ── Minimal fake model ─────────────────────────────────────────────────────────

class _FakeClassifier(nn.Module):
    ordinal = False

    def forward(self, x):
        batch = x.shape[0]
        logits = torch.zeros(batch, 5)
        logits[:, 2] = 5.0
        return logits


class _FakeOrdinalClassifier(nn.Module):
    ordinal = True

    def forward(self, x):
        return torch.ones(x.shape[0], 4) * 5.0


def _rand_pil(h=224, w=224):
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ── _segment_grid ──────────────────────────────────────────────────────────────

def test_segment_grid_shape():
    seg = _segment_grid(224, 224, grid_size=7)
    assert seg.shape == (224, 224)


def test_segment_grid_num_segments():
    seg = _segment_grid(224, 224, grid_size=7)
    assert seg.max() == 48  # 7*7 - 1


def test_segment_grid_all_pixels_assigned():
    seg = _segment_grid(224, 224, grid_size=7)
    assert seg.min() == 0


def test_segment_grid_custom_size():
    seg = _segment_grid(100, 100, grid_size=5)
    assert seg.shape == (100, 100)
    assert seg.max() == 24  # 5*5 - 1


def test_segment_grid_dtype():
    seg = _segment_grid(64, 64, grid_size=4)
    assert seg.dtype == np.int32


# ── lime explain ───────────────────────────────────────────────────────────────

def test_lime_explain_returns_four_tuple():
    model = _FakeClassifier()
    img = _rand_pil()
    result = lime_explain(model, img, val_transform, torch.device("cpu"),
                          grid_size=4, num_samples=20)
    assert isinstance(result, tuple) and len(result) == 4


def test_lime_explain_pred_range():
    model = _FakeClassifier()
    img = _rand_pil()
    pred, conf, importance, segments = lime_explain(
        model, img, val_transform, torch.device("cpu"), grid_size=4, num_samples=20
    )
    assert 0 <= pred <= 4
    assert 0.0 <= conf <= 1.0


def test_lime_explain_importance_shape():
    model = _FakeClassifier()
    img = _rand_pil()
    _, _, importance, _ = lime_explain(
        model, img, val_transform, torch.device("cpu"), grid_size=4, num_samples=20
    )
    assert importance.shape == (224, 224)


def test_lime_explain_importance_range():
    model = _FakeClassifier()
    img = _rand_pil()
    _, _, importance, _ = lime_explain(
        model, img, val_transform, torch.device("cpu"), grid_size=4, num_samples=20
    )
    assert importance.min() >= 0.0
    assert importance.max() <= 1.0 + 1e-6


def test_lime_explain_segments_shape():
    model = _FakeClassifier()
    img = _rand_pil()
    _, _, _, segments = lime_explain(
        model, img, val_transform, torch.device("cpu"), grid_size=4, num_samples=20
    )
    assert segments.shape == (224, 224)


def test_lime_explain_ordinal_model():
    """LIME should work the same way with ordinal models."""
    model = _FakeOrdinalClassifier()
    img = _rand_pil()
    pred, conf, importance, segments = lime_explain(
        model, img, val_transform, torch.device("cpu"), grid_size=3, num_samples=10
    )
    assert 0 <= pred <= 4
    assert importance.shape == (224, 224)


def test_lime_explain_correct_prediction():
    """Model strongly favouring grade 2 should return pred=2."""
    model = _FakeClassifier()
    img = _rand_pil()
    pred, _, _, _ = lime_explain(
        model, img, val_transform, torch.device("cpu"), grid_size=4, num_samples=20
    )
    assert pred == 2
