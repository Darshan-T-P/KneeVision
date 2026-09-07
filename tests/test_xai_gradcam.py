"""Tests for xai/base.py get_prediction, and xai/gradcam.py GradCAM class + explain."""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from kneevision.xai.base import get_prediction
from kneevision.xai.gradcam import GradCAM, explain as gradcam_explain
from kneevision.data.transforms import val_transform


# ── Minimal DenseNet-like model for hook registration ─────────────────────────

class _ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential()
        self.features.denseblock4 = _ConvBlock()

    def forward(self, x):
        return self.features.denseblock4(x)


class _FakeModel(nn.Module):
    """Small model with the same named-module structure GradCAM/ScoreCAM expect."""
    ordinal = False

    def __init__(self):
        super().__init__()
        self.backbone = _FakeBackbone()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 5),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)


class _FakeOrdinalModel(_FakeModel):
    ordinal = True

    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 4),   # ordinal: num_classes - 1 outputs
        )


def _rand_pil(h=224, w=224):
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ── get_prediction ─────────────────────────────────────────────────────────────

def test_get_prediction_returns_tuple():
    model = _FakeModel()
    x = torch.randn(1, 3, 224, 224)
    result = get_prediction(model, x, torch.device("cpu"))
    assert isinstance(result, tuple) and len(result) == 3


def test_get_prediction_softmax_grade_range():
    model = _FakeModel()
    x = torch.randn(1, 3, 224, 224)
    pred, conf, logits = get_prediction(model, x, torch.device("cpu"))
    assert 0 <= pred <= 4
    assert 0.0 <= conf <= 1.0
    assert logits.shape == (1, 5)


def test_get_prediction_ordinal():
    model = _FakeOrdinalModel()
    x = torch.randn(1, 3, 224, 224)
    pred, conf, logits = get_prediction(model, x, torch.device("cpu"))
    assert 0 <= pred <= 4
    assert conf == 1.0  # ordinal always returns 1.0
    assert logits.shape == (1, 4)


def test_get_prediction_deterministic():
    model = _FakeModel()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    p1, c1, _ = get_prediction(model, x, torch.device("cpu"))
    p2, c2, _ = get_prediction(model, x, torch.device("cpu"))
    assert p1 == p2 and abs(c1 - c2) < 1e-6


# ── GradCAM.generate ───────────────────────────────────────────────────────────

def test_gradcam_generate_shape():
    model = _FakeModel()
    gcam = GradCAM(model, target_layer="backbone.features.denseblock4")
    x = torch.randn(1, 3, 224, 224)
    cam = gcam.generate(x)
    assert cam.shape == (224, 224), f"Expected (224,224), got {cam.shape}"


def test_gradcam_cam_range():
    model = _FakeModel()
    gcam = GradCAM(model, target_layer="backbone.features.denseblock4")
    x = torch.randn(1, 3, 224, 224)
    cam = gcam.generate(x)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6


def test_gradcam_explicit_class_idx():
    """generate() with explicit class_idx should not raise."""
    model = _FakeModel()
    gcam = GradCAM(model, target_layer="backbone.features.denseblock4")
    x = torch.randn(1, 3, 224, 224)
    cam = gcam.generate(x, class_idx=0)
    assert cam.shape == (224, 224)


def test_gradcam_cam_not_all_zeros():
    """CAM should have some non-zero structure (informative heatmap)."""
    model = _FakeModel()
    gcam = GradCAM(model, target_layer="backbone.features.denseblock4")
    x = torch.randn(1, 3, 224, 224)
    cam = gcam.generate(x)
    assert cam.max() > 0.0


# ── gradcam_explain ────────────────────────────────────────────────────────────

def test_gradcam_explain_returns_tuple():
    model = _FakeModel()
    img = _rand_pil()
    result = gradcam_explain(model, img, val_transform, torch.device("cpu"))
    assert isinstance(result, tuple) and len(result) == 3


def test_gradcam_explain_overlay_shape():
    model = _FakeModel()
    img = _rand_pil()
    pred, conf, overlay = gradcam_explain(model, img, val_transform, torch.device("cpu"))
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8


def test_gradcam_explain_pred_range():
    model = _FakeModel()
    img = _rand_pil()
    pred, conf, _ = gradcam_explain(model, img, val_transform, torch.device("cpu"))
    assert 0 <= pred <= 4
    assert 0.0 <= conf <= 1.0
