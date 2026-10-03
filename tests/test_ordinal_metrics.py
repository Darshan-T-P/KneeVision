"""Tests for the extended evaluation metrics added during the research audit.

Covers:
- compute_ordinal_metrics(): MAE, within-1/2 accuracy
- compute_calibration_metrics(): ECE, Brier score
- reliability_diagram(): smoke test
- ordinal_error_plot(): smoke test
- Integration with compute_metrics() (backward-compatible)
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kneevision.evaluation.report import (
    compute_calibration_metrics,
    compute_confusion_matrix,
    compute_metrics,
    compute_ordinal_metrics,
    ordinal_error_plot,
    reliability_diagram,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_preds():
    labels = list(range(5)) * 20  # 100 samples, each KL grade 20 times
    return labels, labels[:]  # preds == labels


@pytest.fixture
def off_by_one_preds():
    """All predictions are exactly 1 grade off (shifts grades up, wraps 4→4)."""
    labels = list(range(5)) * 20
    preds = [min(4, l + 1) for l in labels]
    return labels, preds


@pytest.fixture
def random_preds(seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 5, size=200).tolist()
    preds = rng.integers(0, 5, size=200).tolist()
    probs = rng.dirichlet(np.ones(5), size=200)
    return labels, preds, probs


# ── compute_ordinal_metrics ──────────────────────────────────────────────────

class TestComputeOrdinalMetrics:
    def test_perfect_predictions_mae_zero(self, perfect_preds):
        labels, preds = perfect_preds
        m = compute_ordinal_metrics(labels, preds)
        assert m["mae"] == pytest.approx(0.0)

    def test_perfect_predictions_within1_is_1(self, perfect_preds):
        labels, preds = perfect_preds
        m = compute_ordinal_metrics(labels, preds)
        assert m["within_1_accuracy"] == pytest.approx(1.0)
        assert m["within_2_accuracy"] == pytest.approx(1.0)
        assert m["exact_accuracy"] == pytest.approx(1.0)

    def test_off_by_one_mae(self, off_by_one_preds):
        labels, preds = off_by_one_preds
        m = compute_ordinal_metrics(labels, preds)
        # All preds are 1 off (grade 4 stays at 4, so last group has 0 error)
        assert 0.8 <= m["mae"] <= 1.0  # most errors are exactly 1
        assert m["within_1_accuracy"] == pytest.approx(1.0)  # all within 1
        assert m["exact_accuracy"] < 1.0   # not all exact

    def test_off_by_one_within2(self, off_by_one_preds):
        labels, preds = off_by_one_preds
        m = compute_ordinal_metrics(labels, preds)
        assert m["within_2_accuracy"] == pytest.approx(1.0)

    def test_returns_required_keys(self, random_preds):
        labels, preds, _ = random_preds
        m = compute_ordinal_metrics(labels, preds)
        assert "mae" in m
        assert "within_1_accuracy" in m
        assert "within_2_accuracy" in m
        assert "exact_accuracy" in m

    def test_mae_nonnegative(self, random_preds):
        labels, preds, _ = random_preds
        m = compute_ordinal_metrics(labels, preds)
        assert m["mae"] >= 0.0

    def test_within1_gte_exact(self, random_preds):
        labels, preds, _ = random_preds
        m = compute_ordinal_metrics(labels, preds)
        assert m["within_1_accuracy"] >= m["exact_accuracy"]
        assert m["within_2_accuracy"] >= m["within_1_accuracy"]

    def test_within1_range(self, random_preds):
        labels, preds, _ = random_preds
        m = compute_ordinal_metrics(labels, preds)
        assert 0.0 <= m["within_1_accuracy"] <= 1.0

    def test_mae_max_value(self):
        # Max error for 5-class problem is 4 (KL0 predicted as KL4)
        labels = [0] * 50
        preds = [4] * 50
        m = compute_ordinal_metrics(labels, preds)
        assert m["mae"] == pytest.approx(4.0)
        assert m["within_1_accuracy"] == pytest.approx(0.0)
        assert m["exact_accuracy"] == pytest.approx(0.0)


# ── compute_calibration_metrics ──────────────────────────────────────────────

class TestComputeCalibrationMetrics:
    def _perfect_probs(self, labels):
        n = len(labels)
        probs = np.zeros((n, 5))
        for i, l in enumerate(labels):
            probs[i, l] = 1.0
        return probs

    def test_ece_perfect_calibration(self, perfect_preds):
        labels, _ = perfect_preds
        probs = self._perfect_probs(labels)
        m = compute_calibration_metrics(labels, probs)
        # Perfect probs → ECE should be near 0
        assert m["ece"] < 0.05

    def test_brier_score_keys_present(self, random_preds):
        labels, _, probs = random_preds
        m = compute_calibration_metrics(labels, probs)
        for c in range(5):
            assert f"brier_KL{c}" in m
        assert "brier_macro" in m
        assert "ece" in m

    def test_brier_score_range(self, random_preds):
        labels, _, probs = random_preds
        m = compute_calibration_metrics(labels, probs)
        # Brier score is in [0, 1] for binary one-vs-rest
        for c in range(5):
            assert 0.0 <= m[f"brier_KL{c}"] <= 1.0
        assert 0.0 <= m["brier_macro"] <= 1.0

    def test_brier_macro_is_mean_of_classes(self, random_preds):
        labels, _, probs = random_preds
        m = compute_calibration_metrics(labels, probs)
        expected_macro = np.mean([m[f"brier_KL{c}"] for c in range(5)])
        assert m["brier_macro"] == pytest.approx(expected_macro, abs=1e-6)

    def test_ece_range(self, random_preds):
        labels, _, probs = random_preds
        m = compute_calibration_metrics(labels, probs)
        assert 0.0 <= m["ece"] <= 1.0

    def test_uniform_probs_ece(self):
        """Uniform predictions are maximally miscalibrated for frequent classes."""
        labels = [0] * 80 + [1] * 20  # imbalanced
        probs = np.full((100, 5), 0.2)  # all classes equal probability
        m = compute_calibration_metrics(labels, probs)
        # ECE should be non-trivial (class 0 has 80% but predicted 20%)
        assert m["ece"] > 0.0


# ── Integration with compute_metrics ────────────────────────────────────────

class TestComputeMetricsIntegration:
    def test_ordinal_metrics_in_output(self, random_preds):
        labels, preds, probs = random_preds
        m = compute_metrics(labels, preds, probs)
        assert "mae" in m
        assert "within_1_accuracy" in m
        assert "within_2_accuracy" in m
        assert "exact_accuracy" in m

    def test_calibration_metrics_in_output_when_probs_given(self, random_preds):
        labels, preds, probs = random_preds
        m = compute_metrics(labels, preds, probs)
        assert "ece" in m
        assert "brier_macro" in m

    def test_calibration_absent_without_probs(self, random_preds):
        labels, preds, _ = random_preds
        m = compute_metrics(labels, preds)  # no probs
        assert "ece" not in m
        assert "brier_macro" not in m

    def test_backward_compat_existing_keys(self, random_preds):
        """Existing metrics must still be present after the update."""
        labels, preds, probs = random_preds
        m = compute_metrics(labels, preds, probs)
        assert "accuracy" in m
        assert "kappa" in m
        assert "kappa_linear" in m
        assert "macro_f1" in m
        assert "weighted_f1" in m

    def test_all_values_are_floats(self, random_preds):
        labels, preds, probs = random_preds
        m = compute_metrics(labels, preds, probs)
        for k, v in m.items():
            assert isinstance(v, float), f"Key {k!r} is not float: {type(v)}"

    # ── authoritative QWK ────────────────────────────────────────────────────

    def test_qwk_key_present_and_equals_legacy_kappa(self, random_preds):
        labels, preds, _ = random_preds
        m = compute_metrics(labels, preds)
        assert "qwk" in m
        assert m["qwk"] == pytest.approx(m["kappa"])

    def test_qwk_is_quadratic_kappa_of_labels_vs_preds(self):
        labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        preds = [0, 1, 0, 2, 1, 2, 3, 4, 4, 3]
        from sklearn.metrics import cohen_kappa_score
        expected = float(cohen_kappa_score(labels, preds, weights="quadratic"))
        assert compute_ordinal_metrics(labels, preds)["qwk"] == pytest.approx(expected)

    def test_ordinal_metrics_include_qwk(self, random_preds):
        labels, preds, _ = random_preds
        m = compute_ordinal_metrics(labels, preds)
        assert "qwk" in m

    # ── authoritative confusion matrix ───────────────────────────────────────

    def test_confusion_matrix_shape_fixed_by_num_classes(self, random_preds):
        labels, preds, _ = random_preds
        cm = compute_confusion_matrix(labels, preds, num_classes=5)
        assert cm.shape == (5, 5)

    def test_confusion_matrix_does_not_drop_absent_class(self):
        """Without num_classes, an absent trailing class would shrink the matrix."""
        labels = [0, 0, 1, 1, 2, 2]   # KL3/4 never observed
        preds = [0, 1, 1, 2, 2, 2]
        cm = compute_confusion_matrix(labels, preds, num_classes=5)
        assert cm.shape == (5, 5)      # grades 3 and 4 get explicit zero rows/cols
        assert cm[0, 0] == 1 and cm[1, 2] == 1

    def test_confusion_matrix_rows_true_cols_predicted_kl_ordered(self):
        """Explicit row/column semantic contract: rows must be true KL class,
        columns must be predicted KL class, both in KL0..KL4 order."""
        labels = [0, 1, 2, 3, 4, 4]
        preds = [0, 1, 2, 3, 4, 0]
        cm = compute_confusion_matrix(labels, preds, num_classes=5)
        assert cm.shape == (5, 5)
        for i in range(5):
            assert cm[i, i] == 1            # every diagonal true==pred match
        assert cm[4, 0] == 1                # true KL4 predicted KL0
        assert cm.sum(axis=1).tolist() == [1, 1, 1, 1, 2]   # rows = true counts
        assert cm.sum(axis=0).tolist() == [2, 1, 1, 1, 1]   # cols = predicted counts

    def test_within_2_accuracy_exact_on_known_predictions(self):
        """Within-2 = fraction of predictions within two grades of the truth;
        also pins within-1 and MAE on the same known cases."""
        labels = [0, 0, 0, 0]
        preds = [0, 1, 2, 3]             # |errors| = 0,1,2,3
        metrics = compute_ordinal_metrics(labels, preds, num_classes=5)
        assert metrics["within_2_accuracy"] == pytest.approx(0.75)   # 3/4 within 2
        assert metrics["within_1_accuracy"] == pytest.approx(0.5)    # 2/4 within 1
        assert metrics["mae"] == pytest.approx(1.5)
        assert metrics["exact_accuracy"] == pytest.approx(0.25)

    def test_compute_metrics_per_class_never_shrinks_with_num_classes(self):
        labels = [0, 0, 1, 1, 2, 2]
        preds = [0, 1, 1, 2, 2, 2]
        m = compute_metrics(labels, preds, num_classes=5)
        assert "KL 0_precision" in m
        assert "KL 4_precision" in m   # present even though KL4 never appears


# ── Plot smoke tests ─────────────────────────────────────────────────────────

class TestPlots:
    def test_reliability_diagram_returns_figure(self, random_preds):
        import matplotlib.pyplot as plt
        labels, _, probs = random_preds
        fig = reliability_diagram(labels, probs)
        assert fig is not None
        plt.close(fig)

    def test_ordinal_error_plot_returns_figure(self, off_by_one_preds):
        import matplotlib.pyplot as plt
        labels, preds = off_by_one_preds
        fig = ordinal_error_plot(labels, preds)
        assert fig is not None
        plt.close(fig)

    def test_reliability_diagram_saves_to_path(self, random_preds, tmp_path):
        labels, _, probs = random_preds
        out = tmp_path / "reliability.png"
        fig = reliability_diagram(labels, probs, out_path=out)
        assert out.exists()

    def test_ordinal_error_plot_saves_to_path(self, off_by_one_preds, tmp_path):
        labels, preds = off_by_one_preds
        out = tmp_path / "ordinal_error.png"
        fig = ordinal_error_plot(labels, preds, out_path=out)
        assert out.exists()
