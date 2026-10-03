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


# ── known defect: _fix_artifact_root vs MLflow 3.14 ────────────────────────────

class _StaleExperiment:
    """Minimal stand-in for an MLflow `Experiment` with a stale artifact root.

    `mlflow.entities.Experiment.artifact_location` is a read-only property, so the
    mismatch this test needs cannot be produced by mutating a real experiment. Only
    the three attributes `_fix_artifact_root` reads are required.
    """

    def __init__(self, experiment_id, artifact_location, name):
        self.experiment_id = experiment_id
        self.artifact_location = artifact_location
        self.name = name


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MLflow 3.14 removed MlflowClient.update_experiment, which "
        "MLflowTracker._fix_artifact_root still calls (tracking.py:40), so any "
        "experiment with an artifact root from another machine/path raises "
        "AttributeError. strict=True makes the suite FAIL once this is fixed — "
        "delete this test then."
    ),
)
def test_fix_artifact_root_survives_stale_artifact_location():
    """Reproduce the defect left dormant by the pre-existing mlflow.db.

    The stale-root branch is only entered when `artifact_location` differs from
    `PROJECT_ROOT/mlruns/<id>`, which is why a pre-populated database whose
    locations already matched never triggered it.
    """
    from kneevision.utils import tracking as tracking_module

    tracker = MLflowTracker(experiment_name="test_stale_artifact_root")
    stale = _StaleExperiment(
        experiment_id=999,
        artifact_location="/home/someone/else/mlruns/999",
        name="test_stale_artifact_root",
    )

    # Drive the real implementation, bypassing the no-op installed in conftest.py.
    tracking_module.ORIGINAL_FIX_ARTIFACT_ROOT(tracker, stale)


def test_tracker_never_calls_removed_mlflow_api():
    """Documents which removed API the defect above depends on.

    Asserting the *absence* is the point: if a future MLflow release restores
    `MlflowClient.update_experiment`, this flips and the strict xfail above XPASSes,
    which pytest reports as a failure — so neither signal can be missed.
    """
    from mlflow.tracking import MlflowClient

    assert not hasattr(MlflowClient, "update_experiment"), (
        "MlflowClient.update_experiment exists again — remove the strict xfail on "
        "test_fix_artifact_root_survives_stale_artifact_location and validate the "
        "real fix."
    )


# ── run_id property ────────────────────────────────────────────────────────────

def test_run_id_none_before_start():
    tracker = MLflowTracker(experiment_name="test_run_id_exp")
    assert tracker.run_id is None


def test_run_id_set_after_start():
    tracker = MLflowTracker(experiment_name="test_run_id_set_exp")
    tracker.start_run()
    assert isinstance(tracker.run_id, str) and len(tracker.run_id) > 0
    tracker.end_run()
