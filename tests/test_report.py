import numpy as np

from kneevision.evaluation.report import (
    compute_metrics,
    classification_report_text,
    confusion_matrix_plot,
    html_report,
    write_artifacts,
)


def test_compute_metrics_perfect():
    labels = [0, 1, 2, 3, 4] * 10
    metrics = compute_metrics(labels, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["kappa"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["KL 0_precision"] == 1.0


def test_compute_metrics_imperfect():
    labels = [0, 0, 1, 1, 2]
    preds = [0, 1, 1, 1, 2]
    metrics = compute_metrics(labels, preds)
    assert 0 < metrics["accuracy"] < 1.0
    assert 0 < metrics["kappa"] < 1.0
    assert all(0.0 <= metrics[k] <= 1.0 for k in metrics)


def test_compute_metrics_with_probs():
    labels = [0, 1, 2, 3, 4] * 4
    probs = np.eye(5)[labels]
    metrics = compute_metrics(labels, labels, probs=probs)
    assert metrics["KL 0_auc"] == 1.0
    assert metrics["KL 4_auc"] == 1.0


def test_classification_report_text_content():
    text = classification_report_text([0, 0, 1, 1], [0, 1, 1, 0])
    assert "KL 0" in text and "KL 1" in text
    assert "accuracy" in text


def test_confusion_matrix_plot_shape():
    fig = confusion_matrix_plot([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    array = fig.axes[0].images[0].get_array()
    assert array.shape == (5, 5)


def test_html_report_contains_sections(tmp_path):
    fig = confusion_matrix_plot([0, 1, 2], [0, 1, 2])
    html = html_report({"accuracy": 1.0}, fig, "precision   0.9\n")
    assert "<h1>KneeVision Evaluation Report</h1>" in html
    assert "data:image/png;base64," in html
    assert "accuracy" in html


def test_write_artifacts(tmp_path):
    labels = [0, 1, 2, 3, 4] * 4
    preds = labels[:]
    paths = write_artifacts(labels, preds, out_dir=tmp_path)
    assert paths["cm_png"].exists()
    assert paths["report_txt"].exists()
    assert paths["report_html"].exists()
    assert "Confusion Matrix" in paths["report_html"].read_text()
