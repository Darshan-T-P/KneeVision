"""Tests for data/transforms.py — train, val, minority, TTA pipelines."""
import torch
import numpy as np
from PIL import Image

from kneevision.data.transforms import (
    build_val_transform,
    build_minority_transform,
    build_tta_transforms,
    train_transform,
    val_transform,
    minority_transform,
    tta_transforms_list,
)


def _rand_pil(h=300, w=300):
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ── val_transform ──────────────────────────────────────────────────────────────

def test_val_transform_output_shape():
    img = _rand_pil()
    tensor = val_transform(img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_val_transform_is_normalised():
    # After ImageNet normalisation values can be negative
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    tensor = val_transform(img)
    assert tensor.min() < 0, "black image should produce negative values after normalisation"


def test_val_transform_custom_size():
    tfm = build_val_transform(size=128)
    img = _rand_pil()
    assert tfm(img).shape == (3, 128, 128)


# ── train_transform ────────────────────────────────────────────────────────────

def test_train_transform_output_shape():
    img = _rand_pil()
    tensor = train_transform(img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_train_transform_stochastic():
    """Two applications of train_transform to the same image should (almost surely) differ."""
    img = _rand_pil()
    t1 = train_transform(img)
    t2 = train_transform(img)
    assert not torch.allclose(t1, t2), "train_transform should be stochastic"


# ── minority_transform ─────────────────────────────────────────────────────────

def test_minority_transform_output_shape():
    img = _rand_pil()
    tensor = minority_transform(img)
    assert tensor.shape == (3, 224, 224)


def test_minority_transform_custom_size():
    tfm = build_minority_transform(size=96)
    img = _rand_pil()
    assert tfm(img).shape == (3, 96, 96)


# ── TTA transforms ─────────────────────────────────────────────────────────────

def test_tta_list_has_two_transforms():
    assert len(tta_transforms_list) == 2


def test_tta_transforms_output_shape():
    img = _rand_pil()
    for tfm in tta_transforms_list:
        t = tfm(img)
        assert t.shape == (3, 224, 224), f"TTA transform produced wrong shape: {t.shape}"


def test_tta_base_and_flip_differ():
    """Base and horizontal-flip TTA should produce different tensors."""
    img = _rand_pil()
    base, flip = build_tta_transforms()
    t_base = base(img)
    t_flip = flip(img)
    assert not torch.allclose(t_base, t_flip), "Base and flip TTA should differ"


def test_tta_custom_size():
    tfms = build_tta_transforms(size=64)
    img = _rand_pil()
    for tfm in tfms:
        assert tfm(img).shape == (3, 64, 64)
