"""Tests for config/settings.py — environment variable loading and defaults."""

import kneevision.config.settings as settings


# ── basic attribute presence ────────────────────────────────────────────────────

REQUIRED_ATTRS = [
    "DEVICE", "IMAGE_SIZE", "BATCH_SIZE", "NUM_EPOCHS", "LEARNING_RATE",
    "NUM_KL_CLASSES", "CNN_MODEL_NAME", "DATA_DIR", "RAW_DATA_DIR",
    "MODELS_DIR", "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_ENABLED", "TTA_AUGS", "WEIGHT_DECAY", "EARLY_STOP_PATIENCE",
]


def test_all_required_settings_present():
    for attr in REQUIRED_ATTRS:
        assert hasattr(settings, attr), f"Missing setting: {attr}"


# ── type checks ────────────────────────────────────────────────────────────────

def test_image_size_is_int():
    assert isinstance(settings.IMAGE_SIZE, int)


def test_batch_size_is_int():
    assert isinstance(settings.BATCH_SIZE, int)


def test_num_epochs_is_int():
    assert isinstance(settings.NUM_EPOCHS, int)


def test_learning_rate_is_float():
    assert isinstance(settings.LEARNING_RATE, float)


def test_num_kl_classes_is_5():
    assert settings.NUM_KL_CLASSES == 5


def test_tta_augs_is_int():
    assert isinstance(settings.TTA_AUGS, int) and settings.TTA_AUGS >= 1


# ── sanity ranges ──────────────────────────────────────────────────────────────

def test_image_size_reasonable():
    assert 64 <= settings.IMAGE_SIZE <= 512


def test_batch_size_positive():
    assert settings.BATCH_SIZE > 0


def test_num_epochs_positive():
    assert settings.NUM_EPOCHS > 0


def test_learning_rate_in_range():
    assert 1e-6 < settings.LEARNING_RATE < 1.0


def test_weight_decay_non_negative():
    assert settings.WEIGHT_DECAY >= 0.0


def test_early_stop_patience_positive():
    assert settings.EARLY_STOP_PATIENCE > 0


# ── path settings ──────────────────────────────────────────────────────────────

def test_data_dir_is_path():
    from pathlib import Path
    assert isinstance(settings.DATA_DIR, Path)


def test_raw_data_dir_under_data():
    assert settings.RAW_DATA_DIR.parent == settings.DATA_DIR


def test_models_dir_under_project_root():
    assert settings.MODELS_DIR.parent == settings.PROJECT_ROOT


# ── MLflow settings ────────────────────────────────────────────────────────────

def test_mlflow_tracking_uri_is_string():
    assert isinstance(settings.MLFLOW_TRACKING_URI, str)
    assert len(settings.MLFLOW_TRACKING_URI) > 0


def test_mlflow_experiment_name_is_string():
    assert isinstance(settings.MLFLOW_EXPERIMENT_NAME, str)


def test_mlflow_enabled_is_bool():
    assert isinstance(settings.MLFLOW_ENABLED, bool)


# ── model name ─────────────────────────────────────────────────────────────────

def test_cnn_model_name_is_string():
    assert isinstance(settings.CNN_MODEL_NAME, str)
    assert len(settings.CNN_MODEL_NAME) > 0
