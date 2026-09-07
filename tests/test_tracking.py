"""Tests for utils/tracking.py — MLflowTracker lifecycle and helpers."""
import pytest
import mlflow

from kneevision.utils.tracking import MLflowTracker


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_mlflow(tmp_path):
    """Each test gets its own in-memory / temp MLflow tracking URI."""
    uri = f"sqlite:///{tmp_path}/mlflow_test.db"
    mlflow.set_tracking_uri(uri)
    yield
    if mlflow.active_run():
        mlflow.end_run()


# ── MLflowTracker construction ─────────────────────────────────────────────────

def test_tracker_instantiation():
    tracker = MLflowTracker(experiment_name="test_exp")
    assert tracker.experiment_name == "test_exp"
    assert tracker.active_run is None
    assert tracker.run_id is None


def test_tracker_creates_experiment():
    MLflowTracker(experiment_name="test_exp_create")
    experiment = mlflow.get_experiment_by_name("test_exp_create")
    assert experiment is not None


# ── start_run / end_run ────────────────────────────────────────────────────────

def test_start_run_sets_active_run():
    tracker = MLflowTracker(experiment_name="test_run_exp")
    tracker.start_run(run_name="my_run")
    assert tracker.active_run is not None
    assert tracker.run_id is not None
    tracker.end_run()


def test_end_run_clears_active_run():
    tracker = MLflowTracker(experiment_name="test_end_exp")
    tracker.start_run()
    tracker.end_run()
    assert tracker.active_run is None
    assert tracker.run_id is None


def test_start_run_with_tags():
    tracker = MLflowTracker(experiment_name="test_tags_exp")
    tracker.start_run(run_name="tagged", tags={"model": "densenet121", "phase": "2"})
    run_id = tracker.run_id
    # Read via client while run is still active — MLflow 3.x flushes synchronously per call
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    assert run.data.tags.get("model") == "densenet121"
    tracker.end_run()


# ── log_params ─────────────────────────────────────────────────────────────────

def test_log_params():
    tracker = MLflowTracker(experiment_name="test_params_exp")
    tracker.start_run()
    tracker.log_params({"lr": 1e-4, "batch_size": 32})
    run_id = tracker.run_id
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    tracker.end_run()
    assert run.data.params.get("lr") == "0.0001"
    assert run.data.params.get("batch_size") == "32"


def test_log_params_from_settings():
    tracker = MLflowTracker(experiment_name="test_settings_exp")
    tracker.start_run()
    tracker.log_params_from_settings(extra={"model_name": "densenet121"})
    run_id = tracker.run_id
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    tracker.end_run()
    assert "batch_size" in run.data.params
    assert "learning_rate" in run.data.params
    assert run.data.params.get("model_name") == "densenet121"


# ── log_metrics ────────────────────────────────────────────────────────────────

def test_log_metrics():
    tracker = MLflowTracker(experiment_name="test_metrics_exp")
    tracker.start_run()
    tracker.log_metrics({"val_loss": 0.42, "val_kappa": 0.75}, step=1)
    client = mlflow.tracking.MlflowClient()
    history = client.get_metric_history(tracker.run_id, "val_loss")
    assert len(history) == 1
    assert abs(history[0].value - 0.42) < 1e-6
    tracker.end_run()


def test_log_epoch_metrics():
    tracker = MLflowTracker(experiment_name="test_epoch_exp")
    tracker.start_run()
    tracker.log_epoch_metrics(
        epoch=1,
        train_loss=0.9, val_loss=0.8,
        val_kappa=0.6, ema_kappa=0.62,
        lr=1e-3,
    )
    client = mlflow.tracking.MlflowClient()
    assert client.get_metric_history(tracker.run_id, "train_loss")[0].value == pytest.approx(0.9)
    assert client.get_metric_history(tracker.run_id, "val_kappa")[0].value == pytest.approx(0.6)
    tracker.end_run()


# ── set_tag / log_artifact ────────────────────────────────────────────────────

def test_set_tag():
    tracker = MLflowTracker(experiment_name="test_tag_exp")
    tracker.start_run()
    tracker.set_tag("status", "done")
    run_id = tracker.run_id
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    tracker.end_run()
    assert run.data.tags.get("status") == "done"


def test_log_artifact(tmp_path):
    artifact_file = tmp_path / "result.txt"
    artifact_file.write_text("accuracy=0.95")
    tracker = MLflowTracker(experiment_name="test_artifact_exp")
    tracker.start_run()
    tracker.log_artifact(str(artifact_file))  # should not raise
    tracker.end_run()


# ── run_id property ────────────────────────────────────────────────────────────

def test_run_id_none_before_start():
    tracker = MLflowTracker(experiment_name="test_run_id_exp")
    assert tracker.run_id is None


def test_run_id_set_after_start():
    tracker = MLflowTracker(experiment_name="test_run_id_set_exp")
    tracker.start_run()
    assert isinstance(tracker.run_id, str) and len(tracker.run_id) > 0
    tracker.end_run()
