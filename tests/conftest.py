import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import kneevision.utils.tracking as _tracking

# ── Test isolation: keep test runs out of the real experiment store ────────────
# `MLflowTracker.__init__` calls `mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)` and
# `_fix_artifact_root()` pins artifacts to `PROJECT_ROOT/mlruns`. Both re-assert the
# production paths, which defeats the per-test `tmp_path` fixture in
# `tests/test_tracking.py`. The suite was therefore writing ~300 test runs into the
# real `mlflow.db` (10 `test_*` experiments) and real `mlruns/`, contaminating the
# research provenance record.
#
# These are module-level globals, so redirecting them here isolates the suite.
# Application code is deliberately left unmodified, and `settings.PROJECT_ROOT`
# is untouched so dataset-dependent tests still resolve real paths.
_TEST_TMP = Path(tempfile.mkdtemp(prefix="kneevision_test_"))
_tracking.MLFLOW_TRACKING_URI = f"sqlite:///{_TEST_TMP}/mlflow_test.db"
_tracking.PROJECT_ROOT = _TEST_TMP

# Neutralise `_fix_artifact_root` for tests. It calls
# `MlflowClient().update_experiment(...)`, which does not exist in MLflow 3.14
# (see `hasattr(MlflowClient, "update_experiment") == False`). That call is only
# reached when an experiment's `artifact_location` differs from
# `PROJECT_ROOT/mlruns/<id>` — never against the previously polluted production
# DB, where the paths already matched, which is why it went unnoticed. Against a
# fresh test DB it is always reached and raises AttributeError.
#
# This is a LATENT APPLICATION BUG, reported rather than fixed here: on a clean
# checkout with no `mlflow.db`, `MLflowTracker()` raises AttributeError on
# construction. See `tests/test_tracking.py::test_fix_artifact_root_survives_stale_
# artifact_location`, which pins the defect with a strict xfail so the suite turns
# red the moment anyone fixes it.
#
# The unpatched implementation is kept reachable so that test can still drive it.
_tracking.ORIGINAL_FIX_ARTIFACT_ROOT = _tracking.MLflowTracker._fix_artifact_root
_tracking.MLflowTracker._fix_artifact_root = lambda self, experiment: None
