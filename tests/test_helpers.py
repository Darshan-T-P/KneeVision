"""Tests for utils/helpers.py — set_seed, reproducibility, environment capture."""
import random

import numpy as np
import torch

from kneevision.utils.helpers import (
    get_device,
    get_environment_info,
    get_git_commit,
    get_rng_state,
    seed_worker,
    set_rng_state,
    set_seed,
)

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


# ── DataLoader worker seeding (requirement 4) ─────────────────────────────────

def test_seed_worker_gives_distinct_states_per_worker():
    set_seed(42)
    seed_worker(0)
    worker0 = random.random()
    seed_worker(1)
    worker1 = random.random()
    assert worker0 != worker1


def test_seed_worker_is_reproducible_for_same_worker():
    set_seed(7)
    seed_worker(3)
    a = np.random.rand(5)
    set_seed(7)
    seed_worker(3)
    b = np.random.rand(5)
    assert np.allclose(a, b)


# ── RNG snapshot / restore (continuity for resume) ────────────────────────────

def test_rng_state_roundtrip_resumes_sequence():
    set_seed(0)
    state = get_rng_state()
    expected = [random.random() for _ in range(5)]
    # consume RNG so current state diverges
    for _ in range(40):
        random.random()
    set_rng_state(state)
    resumed = [random.random() for _ in range(5)]
    assert resumed == expected


def test_rng_state_includes_python_numpy_torch():
    set_seed(0)
    state = get_rng_state()
    assert "python" in state and "numpy" in state and "torch_cpu" in state


def test_set_rng_state_none_is_noop():
    set_rng_state(None)  # must not raise


# ── environment / version capture ─────────────────────────────────────────────

def test_environment_info_has_required_keys():
    info = get_environment_info()
    for key in ("python_version", "pytorch_version", "torchvision_version",
                "cuda_version", "platform"):
        assert key in info, f"missing {key}"
        assert isinstance(info[key], str)


def test_git_commit_is_short_sha_or_none():
    commit = get_git_commit()
    if commit is None:
        return  # not inside a git repo — acceptable
    assert len(commit) >= 7
    head = commit.split("-")[0]
    assert int(head, 16) >= 0  # hexadecimal short SHA
