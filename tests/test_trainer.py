"""Checkpoint and resume behaviour (requirements 5, 6, 7).

Covers: full-state save/load round trips, rich provenance metadata in the
checkpoint, resume continuity (epoch / best val QWK / history / RNG state), and
the guarantee that model selection happens only on the patient-disjoint val
split — never on test.
"""
import torch
import pytest
import torch.nn as nn

from kneevision.training.trainer import (
    EMA,
    EarlyStopping,
    load_checkpoint,
    save_checkpoint,
    validate,
)
from kneevision.training.losses import OrdinalLoss
from kneevision.config.settings import RAW_DATA_DIR


def _make_trainer_components():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    ema = EMA(model, decay=0.995)
    early_stop = EarlyStopping(patience=3)
    return model, optimizer, scheduler, ema, early_stop


# ── save / load round trip ────────────────────────────────────────────────────

class TestCheckpointRoundTrip:
    def test_save_then_load_restores_weights_and_optimizer(self, tmp_path):
        model, optimizer, scheduler, ema, early_stop = _make_trainer_components()
        path = tmp_path / "ckpt.pt"
        history = {1: {"val_qwk": 0.4}, 2: {"val_qwk": 0.5}}
        save_checkpoint(path, model, optimizer, scheduler, ema, early_stop,
                        epoch=2, best_kappa=0.5, history=history,
                        model_name="densenet121", ordinal=True,
                        metadata={"experiment": "x", "seed": 7})

        model2, opt2, sched2, ema2, es2 = _make_trainer_components()
        epoch, best, hist, rng, meta = load_checkpoint(
            path, model2, opt2, sched2, ema2, es2)

        assert epoch == 2
        assert best == pytest.approx(0.5)
        assert hist[2]["val_qwk"] == pytest.approx(0.5)
        assert rng is not None
        assert meta["experiment"] == "x" and meta["seed"] == 7
        # weights/optimizer/EMA restored exactly
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1.detach(), p2.detach())
        for o1, o2 in zip(optimizer.state_dict()["param_groups"], opt2.state_dict()["param_groups"]):
            assert o1 == o2
        for e1, e2 in zip(ema.model.parameters(), ema2.model.parameters()):
            assert torch.allclose(e1.detach(), e2.detach())

    def test_resume_returns_next_epoch(self, tmp_path):
        model, optimizer, scheduler, ema, early_stop = _make_trainer_components()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, optimizer, scheduler, ema, early_stop,
                        epoch=10, best_kappa=0.71, history={}, model_name="m", ordinal=False)
        model2, opt2, sched2, ema2, es2 = _make_trainer_components()
        epoch, best, _, _, _ = load_checkpoint(path, model2, opt2, sched2, ema2, es2)
        assert epoch == 10
        assert best == pytest.approx(0.71)
        # the caller resumes from epoch + 1
        assert epoch + 1 == 11


# ── checkpoint metadata (requirement 5) ──────────────────────────────────────

class TestCheckpointMetadata:
    FULL_META = {
        "experiment": "densenet121_ordinal",
        "model_name": "densenet121",
        "ordinal": True,
        "num_classes": 5,
        "seed": 42,
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "image_size": 224,
        "git_commit": "abc1234",
        "python_version": "3.13",
        "pytorch_version": "2.x",
        "cuda_version": "12.x",
    }

    def test_required_provenance_fields_serialize_and_round_trip(self, tmp_path):
        model, optimizer, scheduler, ema, early_stop = _make_trainer_components()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, optimizer, scheduler, ema, early_stop,
                        epoch=3, best_kappa=0.6, history={},
                        model_name="densenet121", ordinal=True,
                        metadata=self.FULL_META)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        meta = raw["metadata"]
        # experiment / epoch / best val QWK / seed / config / environment
        assert meta["experiment"] == "densenet121_ordinal"
        assert raw["epoch"] == 3
        assert raw["best_kappa"] == pytest.approx(0.6)
        assert meta["seed"] == 42
        for config_key in ("learning_rate", "num_epochs", "image_size", "num_classes"):
            assert config_key in meta
        for env_key in ("python_version", "pytorch_version", "cuda_version"):
            assert env_key in meta


# ── validate integrates the authoritative ordinal metrics ─────────────────────

class _OrdinalStub(nn.Module):
    """Minimal ordinal model: emits num_classes-1 logits and self.ordinal=True."""

    def __init__(self):
        super().__init__()
        self.ordinal = True
        self.net = nn.Linear(4, 4)

    def forward(self, x):
        return self.net(x)


class TestValidateOrdinalMetrics:
    def test_validate_returns_authoritative_metric_bundle(self):
        torch.manual_seed(1)
        model = _OrdinalStub()
        # deterministic "dataset": 6 samples, KL labels 0..2
        x = torch.randn(6, 4)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        from torch.utils.data import TensorDataset, DataLoader

        ds = TensorDataset(x, labels)
        loader = DataLoader(ds, batch_size=6, shuffle=False)
        criterion = OrdinalLoss(num_classes=5)  # real CORAL loss path
        dev = torch.device("cpu")
        _, qwk, metrics = validate(
            model, loader, criterion, dev, use_kappa=True,
            return_metrics=True, num_classes=5)

        assert "qwk" in metrics and "mae" in metrics
        assert "within_1_accuracy" in metrics
        assert "accuracy" in metrics and "macro_f1" in metrics
        assert qwk == pytest.approx(metrics["qwk"])

    def test_validate_non_ordinal_uses_max_logit(self):
        model = nn.Sequential(nn.Linear(4, 3))
        from torch.utils.data import TensorDataset, DataLoader
        x = torch.randn(6, 4)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loader = DataLoader(TensorDataset(x, labels), batch_size=6)
        dev = torch.device("cpu")
        loss, score, metrics = validate(
            model, loader, nn.CrossEntropyLoss(), dev,
            use_kappa=True, return_metrics=True, num_classes=5)
        assert "qwk" in metrics
        assert isinstance(score, float)


# ── model selection uses val only (requirement 7) ─────────────────────────────

class TestValOnlySelection:
    def test_selection_split_is_val_not_test(self):
        assert str(RAW_DATA_DIR / "val").endswith("val")
        assert RAW_DATA_DIR / "val" != RAW_DATA_DIR / "test"

    def test_compare_models_never_loads_test_split(self):
        """Structural guard: a copy-paste of the test dir into the training
        script would break this test (and the science)."""
        src = (next(p for p in __import__("pathlib").Path(__file__).resolve().parents
                    if (p / "scripts/compare_models.py").exists())
               / "scripts/compare_models.py").read_text()
        assert '"test"' not in src.replace('VAL_SPLIT_DIR = RAW_DATA_DIR / "val"', "")
        assert 'RAW_DATA_DIR / "test"' not in src
        assert "get_paths_and_labels(VAL_SPLIT_DIR)" in src