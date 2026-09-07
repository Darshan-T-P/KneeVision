"""Tests for utils/helpers.py — set_seed and get_device."""
import torch
import numpy as np
import random

from kneevision.utils.helpers import set_seed, get_device


# ── set_seed ───────────────────────────────────────────────────────────────────

def test_set_seed_makes_torch_reproducible():
    set_seed(42)
    t1 = torch.randn(10)
    set_seed(42)
    t2 = torch.randn(10)
    assert torch.allclose(t1, t2), "same seed should produce identical torch tensors"


def test_set_seed_makes_numpy_reproducible():
    set_seed(99)
    a1 = np.random.rand(10)
    set_seed(99)
    a2 = np.random.rand(10)
    assert np.allclose(a1, a2), "same seed should produce identical numpy arrays"


def test_set_seed_makes_random_reproducible():
    set_seed(7)
    r1 = [random.random() for _ in range(10)]
    set_seed(7)
    r2 = [random.random() for _ in range(10)]
    assert r1 == r2, "same seed should produce identical Python random values"


def test_different_seeds_differ():
    set_seed(1)
    t1 = torch.randn(10)
    set_seed(2)
    t2 = torch.randn(10)
    assert not torch.allclose(t1, t2), "different seeds should (almost surely) produce different tensors"


def test_set_seed_default():
    """Calling set_seed() without args should not raise."""
    set_seed()


def test_set_seed_deterministic_flag():
    set_seed(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


# ── get_device ─────────────────────────────────────────────────────────────────

def test_get_device_returns_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_valid_type():
    device = get_device()
    assert device.type in ("cuda", "cpu")


def test_get_device_cuda_matches_availability():
    device = get_device()
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_get_device_tensor_creation():
    """We should be able to create a tensor on the returned device."""
    device = get_device()
    t = torch.tensor([1.0, 2.0]).to(device)
    assert t.device.type == device.type
