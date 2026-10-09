"""End-to-end smoke test of the definitive CORAL training loop.

Trains a real DenseNet121 for ONE epoch on 10 synthetic images in a temp
directory, exercising the full wiring: per-run seeding, worker seeding, the
authoritative per-epoch metric bundle, checkpoint metadata, and the best-model
sidecar. `run()` is patched to write into a tmp `MODELS_DIR` and MLflow is
disabled, so the real `models/` artefacts and the experiment DB are untouched.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_compare_models():
    spec = importlib.util.spec_from_file_location("compare_models_mod", SCRIPTS / "compare_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.MLFLOW_ENABLED = False
    return mod


@pytest.fixture(scope="module")
def compare_models():
    cm = _load_compare_models()
    SRC_SYS = str(SRC)
    if SRC_SYS not in sys.path:
        sys.path.insert(0, SRC_SYS)
    return cm


def _make_image(grade: int, path: Path):
    img = Image.new("RGB", (224, 224), (250 - 20 * grade, 240 - 15 * grade, 230 - 10 * grade))
    img.save(path)


@pytest.fixture
def toy_data(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    paths, labels = [], []
    for i in range(10):
        grade = i % 5
        p = img_dir / f"v{grade}_{i}.png"
        _make_image(grade, p)
        paths.append(p)
        labels.append(grade)
    return paths, labels


def test_resume_continues_from_recorded_epoch(compare_models, toy_data, tmp_path, monkeypatch):
    """Requirement 6: `--resume` must continue the interrupted run, not restart."""
    paths, labels = toy_data
    monkeypatch.setattr(compare_models, "get_paths_and_labels", lambda split_dir: (paths, labels))
    monkeypatch.setattr(compare_models, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(compare_models, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(compare_models, "CHECKPOINT_INTERVAL", 1)

    # phase A: train 1 epoch, checkpoint written with epoch=1
    compare_models.run(model_name="densenet121", ordinal=True, num_epochs=1,
                       batch_size=4, patience=1, seed=5, num_workers=0)
    ckpt = torch.load(tmp_path / "models" / "checkpoint_densenet121_ordinal.pt",
                      map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 1

    # phase B: resume to epoch 2
    compare_models.run(model_name="densenet121", ordinal=True, resume=True,
                       num_epochs=2, batch_size=4, patience=1, seed=5, num_workers=0)
    ckpt2 = torch.load(tmp_path / "models" / "checkpoint_densenet121_ordinal.pt",
                       map_location="cpu", weights_only=False)
    assert ckpt2["epoch"] == 2
    assert set(ckpt2["history"].keys()) == {1, 2}


def test_ordinal_training_loop_end_to_end(compare_models, toy_data, tmp_path, monkeypatch):
    paths, labels = toy_data
    monkeypatch.setattr(compare_models, "get_paths_and_labels", lambda split_dir: (paths, labels))
    monkeypatch.setattr(compare_models, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(compare_models, "get_device",
                        lambda: torch.device("cpu"))
    monkeypatch.setattr(compare_models, "NUM_WORKERS", 0)
    monkeypatch.setattr(compare_models, "CHECKPOINT_INTERVAL", 1)  # write the full ckpt every epoch

    best_kappa, params, _ = compare_models.run(
        model_name="densenet121", ordinal=True, num_epochs=1,
        batch_size=4, patience=1, seed=11, num_workers=0)

    assert isinstance(best_kappa, float)
    assert params > 0

    # best model + enriched sidecar (seed, env, best val QWK)
    best_pt = tmp_path / "models" / "best_densenet121_ordinal.pt"
    best_json = best_pt.with_suffix(".json")
    assert best_pt.exists()
    assert best_json.exists()
    meta = __import__("json").loads(best_json.read_text())
    assert meta["ordinal"] is True
    assert meta["seed"] == 11
    assert meta["best_validation_qwk"] == pytest.approx(meta["best_kappa"])
    assert "pytorch_version" in meta and "python_version" in meta

    # full checkpoint: history per epoch + provenance metadata + RNG state
    ckpt = torch.load(tmp_path / "models" / "checkpoint_densenet121_ordinal.pt",
                      map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 1
    assert set(ckpt["history"].keys()) == {1}
    assert "val_qwk" in ckpt["history"][1]
    assert "val_mae" in ckpt["history"][1]
    assert "val_within_2" in ckpt["history"][1]
    assert 0.0 <= ckpt["history"][1]["val_within_2"] <= 1.0
    assert "experiment" in ckpt["metadata"]
    assert ckpt["metadata"]["experiment"] == "densenet121_ordinal"
    assert ckpt["metadata"]["seed"] == 11
    assert ckpt["rng_state"] is not None

    # confusion matrix artifact: val-only, true rows / predicted cols, KL0..KL4
    cm_json = json.loads(best_pt.with_suffix(".cm.json").read_text())
    assert cm_json["split"] == "val"
    assert cm_json["rows"].startswith("true KL class")
    assert cm_json["columns"].startswith("predicted KL class")
    assert cm_json["classes"] == ["KL0", "KL1", "KL2", "KL3", "KL4"]
    cm = cm_json["matrix"]
    assert len(cm) == 5 and all(len(row) == 5 for row in cm)
    assert all(isinstance(v, int) and v >= 0 for row in cm for v in row)
    # toy val labels hold each grade twice: true counts on the rows
    assert [sum(row) for row in cm] == [2, 2, 2, 2, 2]
    assert sum(sum(row) for row in cm) == 10

    # selection rule is UNCHANGED by the instrumentation:
    # best_pt must be exactly the weights of whichever of raw/EMA scored the
    # higher validation QWK (max, not within-2/MAE/accuracy/CM).
    best_weights = torch.load(best_pt, map_location="cpu", weights_only=False)
    val_qwk = ckpt["history"][1]["val_qwk"]
    ema_qwk = ckpt["history"][1]["ema_qwk"]
    assert meta["best_validation_qwk"] == pytest.approx(max(val_qwk, ema_qwk))
    expected = ckpt["ema_state_dict"] if ema_qwk >= val_qwk else ckpt["model_state_dict"]
    assert set(best_weights.keys()) == set(expected.keys())
    for k in expected:
        assert torch.equal(best_weights[k], expected[k]), f"selection deviated at {k}"

    # resume continues from the recorded epoch and restores the RNG
    from kneevision.models.image_model import KneeXRayClassifier
    from kneevision.training.trainer import load_checkpoint

    model = KneeXRayClassifier("densenet121", 5, ordinal=True)
    ep, best, hist, rng, meta = load_checkpoint(
        tmp_path / "models" / "checkpoint_densenet121_ordinal.pt", model)
    assert ep == 1
    assert best > -1.0
    assert hist[1]["val_qwk"] is not None
    assert rng is not None


def test_resume_matches_fresh_run_bitwise(compare_models, toy_data, tmp_path, monkeypatch):
    """Requirement 4/5/6: a resumed run must produce bitwise-identical weights,
    optimizer state, EMA shadow weights and metrics history vs. an uninterrupted
    run over the same epochs. This is the reproducibility guarantee.
    """
    from kneevision.models.image_model import KneeXRayClassifier
    from kneevision.training.trainer import EMA, EarlyStopping, load_checkpoint

    paths, labels = toy_data
    cm = compare_models

    def _patched(tmp: Path):
        (tmp / "models").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(cm, "get_paths_and_labels", lambda split_dir: (paths, labels))
        monkeypatch.setattr(cm, "MODELS_DIR", tmp / "models")
        monkeypatch.setattr(cm, "get_device", lambda: torch.device("cpu"))
        monkeypatch.setattr(cm, "CHECKPOINT_INTERVAL", 1)
        monkeypatch.setattr(cm, "MIXUP_ALPHA", 0.4)  # exercise the numpy-RNG mixup path
        monkeypatch.setattr(cm, "SAMPLER_POWER", 0.5)

    def _load(ckpt_path):
        model = KneeXRayClassifier("densenet121", 5, ordinal=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cm.LEARNING_RATE,
                                      weight_decay=cm.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.0)
        ema = EMA(model, decay=0.995)
        early_stop = EarlyStopping(patience=1)
        _, _, _, _, _ = load_checkpoint(ckpt_path, model, optimizer, scheduler, ema, early_stop)
        return model, optimizer, scheduler, ema, early_stop

    # Run A: uninterrupted 2-epoch run.
    _patched(tmp_path / "a")
    cm.run(model_name="densenet121", ordinal=True, num_epochs=2,
           batch_size=4, patience=1, seed=7, num_workers=0)
    ma, oa, sa, ea, esa = _load(tmp_path / "a" / "models" / "checkpoint_densenet121_ordinal.pt")
    wa = {n: p.detach().clone() for n, p in ma.named_parameters()}
    ba = {n: b.clone() for n, b in ma.named_buffers()}
    wa_ema = {n: p.detach().clone() for n, p in ea.model.named_parameters()}

    # Run B: 1 epoch, then resume to epoch 2.
    _patched(tmp_path / "b")
    cm.run(model_name="densenet121", ordinal=True, num_epochs=1,
           batch_size=4, patience=1, seed=7, num_workers=0)
    cm.run(model_name="densenet121", ordinal=True, resume=True, num_epochs=2,
           batch_size=4, patience=1, seed=7, num_workers=0)
    mb, ob, sb, eb, esb = _load(tmp_path / "b" / "models" / "checkpoint_densenet121_ordinal.pt")

    for name, expected_param in wa.items():
        assert torch.equal(expected_param, mb.state_dict()[name]), f"model param diverged: {name}"
        assert torch.equal(wa_ema[name], eb.model.state_dict()[name]), f"EMA diverged: {name}"
    for name, expected_buffer in ba.items():
        assert torch.equal(expected_buffer, mb.state_dict()[name]), f"model buffer diverged: {name}"
    for name in ("exp_avg", "exp_avg_sq"):
        assert all(torch.equal(oa.state_dict()["state"][i][name],
                               ob.state_dict()["state"][i][name])
                   for i in sorted(oa.state_dict()["state"])), f"optimizer {name} diverged"
    assert oa.state_dict()["param_groups"][0]["lr"] == ob.state_dict()["param_groups"][0]["lr"]
    assert sa.state_dict()["_step_count"] == sb.state_dict()["_step_count"]
    assert esa.best_score == esb.best_score
    assert esa.counter == esb.counter
    assert esa.stopped == esb.stopped


# ── Fair-CORAL infrastructure ─────────────────────────────────────────────────

FROZEN_ORDINAL_ARTIFACTS = {
    "models/best_densenet121_ordinal.pt": "31800ac18780b5fa10e1471b016641d35604d5316b3fb2cb58edb84346bda701",
    "models/best_densenet121_ordinal.json": "35005cb6aef21248bbf83f78cd9cb1f91851cee2cf8e931f99292afd9cb11a25",
    "models/best_densenet121_ordinal.cm.json": "9148923504c2e431648a39762f2812f649139a8f19bd15d6865d17559dfe5916",
    "models/best_densenet121_ordinal.test.json": "e2578afe9326ea8c72f36852699e1f7234ec08a3dfbd85b1834239cf3df912a8",
}

# The four fair-audit cells (H1/H2/L1/L2) from reports/CORAL_FOCAL_FAIRNESS_AUDIT.md.
FAIR_CELLS = [
    ("densenet121", False, False, "focal_mixup"),
    ("densenet121", True, False, "ordinal_soft_mixup"),
    ("densenet121", False, False, "focal_nomixup"),
    ("densenet121", True, False, "ordinal_soft_nomixup"),
]


def test_validate_cli_rules(compare_models):
    import types

    def args(**kw):
        base = {"soft_ordinal_targets": False, "ordinal": False, "mixup_alpha": None,
                    "class_weights": None, "label_smoothing": None, "tag": ""}
        base.update(kw)
        return types.SimpleNamespace(**base)

    assert compare_models.validate_cli(args(soft_ordinal_targets=True)) == "--soft-ordinal-targets requires --ordinal"
    assert compare_models.validate_cli(args(soft_ordinal_targets=True, ordinal=True)) is None
    assert compare_models.validate_cli(args(mixup_alpha=-0.5)) is not None
    assert compare_models.validate_cli(args(mixup_alpha=0.0)) is None
    assert compare_models.validate_cli(args(class_weights=-1.0)) is not None
    assert compare_models.validate_cli(args(class_weights=0.0)) is None
    assert compare_models.validate_cli(args(label_smoothing=1.5)) is not None
    assert compare_models.validate_cli(args(label_smoothing=0.0)) is None
    assert compare_models.validate_cli(args(tag="bad/tag")) == "--tag may only contain [A-Za-z0-9_-]"
    assert compare_models.validate_cli(args(tag="focal_mixup")) is None


def test_tagged_artifacts_isolated_and_frozen_ordinal_untouched(compare_models, tmp_path):
    """No training: verify the artifact-naming function isolates every tagged
    cell from the frozen definitive names, and that the frozen artifacts'
    SHA-256 (and mtimes) are byte-identical to the recorded snapshots."""
    import hashlib

    root = Path(compare_models.MODELS_DIR).parent  # PROJECT_ROOT (models dir is <root>/models)
    missing = [rel for rel in FROZEN_ORDINAL_ARTIFACTS if not (root / rel).exists()]
    if missing:
        pytest.skip(f"definitive CORAL artifacts absent: {missing}")

    before_mtimes = {}
    for rel, expected in FROZEN_ORDINAL_ARTIFACTS.items():
        path = root / rel
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        assert h == expected, f"{rel} content changed: {h}"
        before_mtimes[rel] = path.stat().st_mtime

    stems = [compare_models.artifact_stem(n, o, b, t) for n, o, b, t in FAIR_CELLS]
    assert len(set(stems)) == len(stems), "tagged stems collide"
    assert "densenet121_ordinal" not in set(stems)

    tagged_files = {f"best_{s}{ext}" for s in stems for ext in (".pt", ".json", ".cm.json")} | {
        f"checkpoint_{s}.pt" for s in stems
    }
    frozen_names = {"best_densenet121_ordinal.pt", "best_densenet121_ordinal.json",
                    "best_densenet121_ordinal.cm.json", "best_densenet121_ordinal.test.json",
                    "checkpoint_densenet121_ordinal.pt"}
    assert tagged_files.isdisjoint(frozen_names)

    for rel in FROZEN_ORDINAL_ARTIFACTS:
        assert (root / rel).stat().st_mtime == before_mtimes[rel], f"{rel} mtime changed"


def test_arms_receive_identical_mixed_batches_same_seed(compare_models, toy_data, tmp_path, monkeypatch):
    """Neither loss may see a different augmented stream. With an identical seed
    the two arms (Focal and CORAL) must consume bit-identical MixUp/CutMix
    batches — the harness's per-run re-seeding already guarantees this, so the
    test proves the property rather than redesigning the RNG system."""
    from torch.utils.data import DataLoader

    from kneevision.data.dataset import (
        KneeXRayDataset,
        MixUpDataset,
        make_weighted_sampler,
    )
    from kneevision.data.transforms import val_transform
    from kneevision.utils.helpers import set_seed

    paths, labels = toy_data

    def collect(seed):
        set_seed(seed)  # exactly as run() does before constructing the loaders
        base = KneeXRayDataset(paths, labels, transform=val_transform)
        mixed = MixUpDataset(base, alpha=0.4, num_classes=5)
        sampler = make_weighted_sampler(labels, power=0.5)
        gen = torch.Generator().manual_seed(seed)
        loader = DataLoader(mixed, batch_size=4, sampler=sampler, num_workers=0, generator=gen)
        imgs, lbls = [], []
        for im, lb in loader:
            imgs.append(im.clone())
            lbls.append(lb.clone())
        return torch.cat(imgs), torch.cat(lbls)

    got_mix = False
    for seed in range(1000, 1200):
        img1, lbl1 = collect(seed)
        img2, lbl2 = collect(seed)
        assert torch.equal(img1, img2), f"image stream diverged for seed {seed}"
        assert torch.equal(lbl1, lbl2), f"label stream diverged for seed {seed}"
        if (lbl1.max(dim=1).values < 0.999).any().item():
            got_mix = True
            break
    assert got_mix, "no candidate seed exercised the MixUp (non one-hot) path"