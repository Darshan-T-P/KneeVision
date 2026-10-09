import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_algorithms: bool = False) -> None:
    """Seed Python, NumPy, torch, and every CUDA device.

    `deterministic_algorithms=True` additionally enables
    ``torch.use_deterministic_algorithms`` (which raises rather than silently
    producing a non-deterministic result) and sets ``CUBLAS_WORKSPACE_CONFIG`` so
    cuBLAS GEMMs are reproducible. Off by default because several stochastic ops
    have no deterministic implementation and would raise.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic_algorithms:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: give each worker a distinct, stable seed.

    Pass this (plus a ``generator``, and ``persistent_workers`` if you can afford
    them) to every ``DataLoader`` so a seeded run produces the same batch sequence
    independent of the worker pool shape.
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_rng_state() -> dict[str, Any]:
    """Snapshot the full process RNG state for bitwise-continuous resume."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore a snapshot produced by :func:`get_rng_state`."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].clone())
    cuda_states = [s.clone() for s in (state.get("torch_cuda") or [])]
    if cuda_states and torch.cuda.is_available():
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            torch.cuda.set_rng_state(cuda_states[0])


def get_git_commit(repo_root: Path | None = None) -> str | None:
    """Best-effort short SHA + dirty flag, or ``None`` outside a git repo."""
    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip() != ""
    except (subprocess.SubprocessError, OSError):
        return None
    return f"{sha}{'-dirty' if dirty else ''}"


def get_environment_info() -> dict[str, str]:
    """Capture the runtime/environment facts that belong in every checkpoint."""
    info: dict[str, str] = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    try:
        import torchvision
        info["torchvision_version"] = torchvision.__version__
    except ImportError:
        info["torchvision_version"] = "not installed"
    info["cuda_version"] = torch.version.cuda or "cpu"
    info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.version() else "n/a"
    info["gpu_name"] = (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else "none")
    info["platform"] = str(platform.platform())
    return info


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")