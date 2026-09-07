"""Tests for data/dataset.py — KneeXRayDataset, MixUpDataset, make_weighted_sampler."""
import numpy as np
import pytest
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler

from kneevision.data.dataset import KneeXRayDataset, MixUpDataset, make_weighted_sampler
from kneevision.data.transforms import val_transform, build_val_transform


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_png(path: Path, colour=(128, 64, 32)):
    img = Image.new("RGB", (64, 64), colour)
    img.save(path)


def _make_dataset(tmp_path: Path, n=6):
    paths, labels = [], []
    for i in range(n):
        p = tmp_path / f"img_{i}.png"
        _make_png(p)
        paths.append(p)
        labels.append(i % 5)
    return paths, labels


# ── KneeXRayDataset ────────────────────────────────────────────────────────────

def test_dataset_len(tmp_path):
    paths, labels = _make_dataset(tmp_path, n=8)
    ds = KneeXRayDataset(paths, labels, transform=val_transform)
    assert len(ds) == 8


def test_dataset_getitem_shape(tmp_path):
    paths, labels = _make_dataset(tmp_path, n=4)
    ds = KneeXRayDataset(paths, labels, transform=val_transform)
    img, label = ds[0]
    assert img.shape == (3, 224, 224)
    assert isinstance(label, int)


def test_dataset_label_passthrough(tmp_path):
    paths, labels = _make_dataset(tmp_path, n=5)
    ds = KneeXRayDataset(paths, labels, transform=val_transform)
    for i, (_, lbl) in enumerate(ds):
        assert lbl == labels[i]


def test_dataset_minority_transform_applied(tmp_path):
    """Minority-class images should use the minority transform, not the standard one."""
    paths, labels = _make_dataset(tmp_path, n=4)
    # All images are grade 0 — mark 0 as minority
    labels = [0] * 4

    # Standard transform → 224×224; minority → 96×96 for easy distinction
    std_tfm = build_val_transform(size=224)
    min_tfm = build_val_transform(size=96)

    ds = KneeXRayDataset(
        paths, labels,
        transform=std_tfm,
        minority_transform=min_tfm,
        minority_labels={0},
    )
    img, _ = ds[0]
    assert img.shape == (3, 96, 96), "minority image should use minority_transform"


def test_dataset_no_transform_returns_pil(tmp_path):
    """Without any transform the dataset should still not crash."""
    paths, labels = _make_dataset(tmp_path, n=2)
    ds = KneeXRayDataset(paths, labels)
    img, label = ds[0]
    assert isinstance(img, Image.Image)


def test_dataset_dataloader_batch(tmp_path):
    paths, labels = _make_dataset(tmp_path, n=6)
    ds = KneeXRayDataset(paths, labels, transform=val_transform)
    loader = DataLoader(ds, batch_size=3)
    batch_imgs, batch_labels = next(iter(loader))
    assert batch_imgs.shape == (3, 3, 224, 224)
    assert batch_labels.shape == (3,)


# ── MixUpDataset ───────────────────────────────────────────────────────────────

def test_mixup_dataset_len(tmp_path):
    paths, labels = _make_dataset(tmp_path, n=10)
    base = KneeXRayDataset(paths, labels, transform=val_transform)
    mixed = MixUpDataset(base, alpha=0.4, num_classes=5)
    assert len(mixed) == 10


def test_mixup_output_shape(tmp_path):
    paths, labels = _make_dataset(tmp_path, n=8)
    base = KneeXRayDataset(paths, labels, transform=val_transform)
    mixed = MixUpDataset(base, alpha=0.4, num_classes=5)
    img, lbl = mixed[0]
    assert img.shape == (3, 224, 224)
    assert lbl.shape == (5,)


def test_mixup_label_sums_to_one(tmp_path):
    """MixUp soft labels must sum to 1 (convex combination)."""
    paths, labels = _make_dataset(tmp_path, n=10)
    base = KneeXRayDataset(paths, labels, transform=val_transform)
    mixed = MixUpDataset(base, alpha=0.4, num_classes=5)
    for i in range(len(mixed)):
        _, lbl = mixed[i]
        assert abs(lbl.sum().item() - 1.0) < 1e-5, f"sample {i}: soft label sum ≠ 1"


def test_mixup_alpha_zero_returns_onehot(tmp_path):
    """With alpha=0 MixUp is skipped — labels should be one-hot."""
    paths, labels = _make_dataset(tmp_path, n=6)
    labels = [2] * 6  # fixed grade for determinism
    base = KneeXRayDataset(paths, labels, transform=val_transform)
    mixed = MixUpDataset(base, alpha=0.0, num_classes=5)
    for i in range(len(mixed)):
        _, lbl = mixed[i]
        # one-hot: one entry = 1, rest = 0
        assert lbl.sum().item() == pytest.approx(1.0)
        assert lbl.max().item() == pytest.approx(1.0)


# ── make_weighted_sampler ──────────────────────────────────────────────────────

def test_weighted_sampler_returns_correct_type():
    labels = [0, 0, 0, 1, 1, 2]
    sampler = make_weighted_sampler(labels)
    assert isinstance(sampler, WeightedRandomSampler)


def test_weighted_sampler_num_samples():
    labels = [0, 1, 2, 3, 4] * 4
    sampler = make_weighted_sampler(labels)
    assert sampler.num_samples == len(labels)


def test_weighted_sampler_minority_gets_higher_weight():
    """Minority class (fewer samples) should get a higher sampling weight."""
    labels = [0] * 90 + [4] * 10  # class 4 is minority
    sampler = make_weighted_sampler(labels, power=1.0)
    weights = sampler.weights.numpy()
    # Weight for a class-4 sample > weight for a class-0 sample
    w_majority = weights[0]   # first 90 are class 0
    w_minority = weights[-1]  # last 10 are class 4
    assert w_minority > w_majority


def test_weighted_sampler_power_zero_gives_uniform():
    """power=0 → all weights equal → uniform sampling."""
    labels = [0] * 9 + [1] * 1
    sampler = make_weighted_sampler(labels, power=0.0)
    w = sampler.weights.numpy()
    assert np.allclose(w, w[0]), "power=0 should yield uniform weights"
