"""Reusable evaluation metrics and report generation (confusion matrix, HTML/TXT)."""

import base64
import io
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

KL_GRADE_NAMES = ["KL 0", "KL 1", "KL 2", "KL 3", "KL 4"]
KL_GRADE_DESCRIPTIONS = {
    0: "Normal",
    1: "Doubtful",
    2: "Mild",
    3: "Moderate",
    4: "Severe",
}


def _as_arrays(labels, preds):
    return np.asarray(labels), np.asarray(preds)


def resolve_num_classes(labels, preds, num_classes: int | None = None) -> int:
    """Resolve the grading scale width, `num_classes` always winning when given.

    Inference from the observed maximum silently drops trailing grades that never
    appear in either split (rare KL4 predictions on a val fold, for example), which
    would shorten the per-class arrays and misalign the confusion matrix. Any
    definitive run should pass `num_classes` explicitly.
    """
    if num_classes is not None:
        return int(num_classes)
    labels, preds = _as_arrays(labels, preds)
    if labels.size == 0 and preds.size == 0:
        return 0
    return int(max(labels.max(initial=0), preds.max(initial=0))) + 1


def n_classes(labels, preds, num_classes: int | None = None) -> int:
    return resolve_num_classes(labels, preds, num_classes)


def compute_ordinal_metrics(labels, preds, num_classes: int | None = None) -> dict[str, float]:
    """Compute ordinal-aware metrics for KL grading.

    This is the authoritative ordinal metric block; `compute_metrics` delegates to
    it rather than re-deriving anything. These are the clinically most relevant
    metrics for an ordinal regression task: a prediction that is off by one grade
    is far less dangerous than one off by three. Quadratic kappa already captures
    this, but explicit within-k accuracy and MAE are more interpretable in
    clinical contexts.
    """
    labels, preds = _as_arrays(labels, preds)
    errors = np.abs(labels.astype(int) - preds.astype(int))
    return {
        "qwk": float(cohen_kappa_score(labels, preds, weights="quadratic")),
        "mae": float(errors.mean()),
        "within_1_accuracy": float((errors <= 1).mean()),
        "within_2_accuracy": float((errors <= 2).mean()),
        "exact_accuracy": float((errors == 0).mean()),
    }


def compute_calibration_metrics(labels, probs, n_bins: int = 10) -> dict[str, float]:
    """Compute Expected Calibration Error (ECE) and per-class Brier scores.

    Calibration is separate from accuracy: a model can be accurate but
    systematically overconfident (bad for downstream clinical decision-making).
    ECE measures the average gap between predicted confidence and empirical accuracy.
    """
    labels_arr = np.asarray(labels, dtype=int)
    probs_arr = np.asarray(probs)  # shape [N, C]
    n, n_classes = probs_arr.shape

    # ECE over all classes (macro-average of per-class ECE)
    ece_per_class = []
    for c in range(n_classes):
        c_probs = probs_arr[:, c]
        c_labels = (labels_arr == c).astype(float)
        # Bin by predicted probability
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece_c = 0.0
        n_edges = len(bin_edges)
        for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            # Close the last bin on both ends so prob == 1.0 (routine after
            # softmax saturation or ordinal_to_probs's clamp/renormalize)
            # lands in a bin instead of being silently dropped from ECE.
            if i == n_edges - 2:
                mask = (c_probs >= lo) & (c_probs <= hi)
            else:
                mask = (c_probs >= lo) & (c_probs < hi)
            if mask.sum() == 0:
                continue
            acc = c_labels[mask].mean()
            conf = c_probs[mask].mean()
            ece_c += mask.sum() / n * abs(acc - conf)
        ece_per_class.append(ece_c)

    metrics = {"ece": float(np.mean(ece_per_class))}

    # Brier score per class (lower is better, 0=perfect)
    for c in range(n_classes):
        c_labels = (labels_arr == c).astype(float)
        metrics[f"brier_KL{c}"] = float(brier_score_loss(c_labels, probs_arr[:, c]))
    # NOTE: "brier_macro" is the MEAN of the C one-vs-rest Brier scores, i.e.
    # (1/(N*C)) * sum_i sum_c (p_ic - y_ic)^2. That is a defensible quantity
    # but it is NOT the textbook multiclass Brier score, which SUMS (not
    # averages) over classes: (1/N) * sum_i sum_c (p_ic - y_ic)^2 = C * brier_macro.
    # Reporting brier_macro unlabeled next to literature Brier numbers makes
    # the model look ~C times better calibrated than it is by that other
    # convention, so both are exposed here under unambiguous names.
    metrics["brier_macro"] = float(np.mean([metrics[f"brier_KL{c}"] for c in range(n_classes)]))
    metrics["brier_multiclass"] = float(n_classes) * metrics["brier_macro"]

    return metrics


def compute_confusion_matrix(labels, preds, class_names=None,
                             num_classes: int | None = None) -> np.ndarray:
    """The single authoritative confusion matrix for this project.

    Rows are true grades, columns are predicted grades, and the matrix is always
    shaped `(num_classes, num_classes)` so axis meaning never shifts between the
    training-time log, the test report, and the exported figure. Every other
    confusion-matrix consumer in the repo delegates here.
    """
    labels, preds = _as_arrays(labels, preds)
    n = resolve_num_classes(labels, preds, num_classes)
    return confusion_matrix(labels, preds, labels=list(range(n)))


def compute_metrics(labels, preds, probs=None, class_names=None,
                     num_classes: int | None = None) -> dict[str, float]:
    """Return a flat dict of scalar metrics ready for mlflow.log_metrics.

    The authoritative metric bundle. Includes: QWK (quadratic kappa, also exposed
    under the legacy `kappa` key), linear kappa, accuracy, MAE, within-1/2
    accuracy, macro/weighted F1, per-class precision/recall/F1, and — when `probs`
    is supplied — per-class AUC, ECE and Brier score.

    Scalars only: MLflow's `log_metrics` cannot accept arrays, so the confusion
    matrix is exposed separately via `compute_confusion_matrix`.
    """
    labels, preds = _as_arrays(labels, preds)
    n = resolve_num_classes(labels, preds, num_classes)
    class_names = class_names or [f"KL {i}" for i in range(n)]

    # Ordinal block (QWK, MAE, within-k) comes from the authoritative helper.
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "kappa_linear": float(cohen_kappa_score(labels, preds, weights="linear")),
    }
    metrics.update(compute_ordinal_metrics(labels, preds, num_classes=num_classes))
    # Legacy alias retained: MLflow runs and existing dashboards key on "kappa".
    metrics["kappa"] = metrics["qwk"]

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, labels=list(range(n)), zero_division=0)
    metrics["macro_precision"] = float(p.mean())
    metrics["macro_recall"] = float(r.mean())
    metrics["macro_f1"] = float(f1.mean())
    metrics["weighted_f1"] = float(
        precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)[2]
    )

    for i, name in enumerate(class_names):
        metrics[f"{name}_precision"] = float(p[i])
        metrics[f"{name}_recall"] = float(r[i])
        metrics[f"{name}_f1"] = float(f1[i])

    if probs is not None and np.asarray(probs).shape[1] == n:
        one_hot = np.eye(n)[labels]
        for i, name in enumerate(class_names):
            try:
                metrics[f"{name}_auc"] = float(roc_auc_score(one_hot[:, i], np.asarray(probs)[:, i]))
            except ValueError:
                continue
        # Calibration metrics
        metrics.update(compute_calibration_metrics(labels, np.asarray(probs)))

    return {k: float(v) for k, v in metrics.items()}


def classification_report_text(labels, preds, class_names=None, digits: int = 4,
                               num_classes: int | None = None) -> str:
    labels, preds = _as_arrays(labels, preds)
    n = resolve_num_classes(labels, preds, num_classes)
    class_names = class_names or [f"KL {i}" for i in range(n)]
    return classification_report(labels, preds, target_names=class_names, digits=digits, zero_division=0)


def reliability_diagram(labels, probs, class_names=None, n_bins: int = 10,
                        out_path: Path | None = None) -> plt.Figure:
    """Plot per-class reliability diagrams (calibration curves).

    A well-calibrated model's curve should follow the diagonal. Curves above
    the diagonal = underconfident; below = overconfident.
    """
    labels_arr = np.asarray(labels, dtype=int)
    probs_arr = np.asarray(probs)
    n_classes = probs_arr.shape[1]
    class_names = class_names or [f"KL {i}" for i in range(n_classes)]
    colors = ["#0284c7", "#7c3aed", "#0d9488", "#f59e0b", "#ef4444"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for c, name in enumerate(class_names):
        c_probs = probs_arr[:, c]
        c_labels = (labels_arr == c).astype(float)
        accs = []
        n_edges = len(bin_edges)
        for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if i == n_edges - 2:
                mask = (c_probs >= lo) & (c_probs <= hi)
            else:
                mask = (c_probs >= lo) & (c_probs < hi)
            accs.append(c_labels[mask].mean() if mask.sum() > 0 else np.nan)
        ax.plot(bin_centers, accs, marker="o", ms=5,
                color=colors[c % len(colors)], label=name)

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction Positive")
    ax.set_title("Reliability Diagram (Calibration)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.4)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def ordinal_error_plot(labels, preds, out_path: Path | None = None) -> plt.Figure:
    """Plot the distribution of |true_KL - predicted_KL| errors."""
    labels_arr = np.asarray(labels, dtype=int)
    preds_arr = np.asarray(preds, dtype=int)
    errors = np.abs(labels_arr - preds_arr)
    max_err = int(errors.max())
    counts = [(errors == e).sum() for e in range(max_err + 1)]
    total = len(errors)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(max_err + 1), [c / total * 100 for c in counts],
                  color=["#0d9488", "#0284c7", "#f59e0b", "#ef4444", "#7c3aed"][:max_err + 1],
                  edgecolor="black", linewidth=0.8)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{count}\n({count/total*100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("|True KL − Predicted KL|")
    ax.set_ylabel("% of Test Cases")
    ax.set_title("Ordinal Error Distribution (Clinical Safety)")
    ax.set_xticks(range(max_err + 1))
    within1 = sum(counts[:2]) / total * 100
    ax.axvline(x=1.5, color="red", linestyle="--", lw=1.5, label=f"Within±1: {within1:.1f}%")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def confusion_matrix_plot(labels, preds, class_names=None, out_path: Path | None = None,
                          num_classes: int | None = None) -> plt.Figure:
    labels, preds = _as_arrays(labels, preds)
    n = resolve_num_classes(labels, preds, num_classes)
    class_names = class_names or [f"KL {i}" for i in range(n)]
    cm = compute_confusion_matrix(labels, preds, num_classes=n)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), class_names, rotation=45, ha="right")
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _metrics_table_html(metrics: dict[str, float]) -> str:
    rows = []
    for key in sorted(metrics):
        value = metrics[key]
        formatted = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
        rows.append(f"<tr><td>{key}</td><td>{formatted}</td></tr>")
    return "<table><tr><th>Metric</th><th>Value</th></tr>" + "".join(rows) + "</table>"


def html_report(metrics: dict[str, float], cm_fig: plt.Figure, report_txt: str,
                extra_notes: str = "") -> str:
    """Self-contained HTML report with the metric table and embedded confusion matrix."""
    cm_img = _fig_to_base64(cm_fig)
    report_html = "<br>".join(line for line in report_txt.splitlines() if line.strip())
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>KneeVision Evaluation Report</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; margin: 16px 0; }}
td, th {{ border: 1px solid #ccc; padding: 4px 10px; text-align: right; }}
th {{ background: #f0f0f0; }}
h2 {{ margin-top: 32px; }}
pre {{ background: #f7f7f7; padding: 12px; border-radius: 4px; overflow-x: auto; }}
</style></head>
<body>
<h1>KneeVision Evaluation Report</h1>
{extra_notes}
<h2>Metrics</h2>
{_metrics_table_html(metrics)}
<h2>Confusion Matrix</h2>
<img src="data:image/png;base64,{cm_img}" alt="confusion matrix" style="max-width:100%"/>
<h2>Classification Report</h2>
<pre>{report_html}</pre>
</body>
</html>
"""


def write_artifacts(labels, preds, probs=None, out_dir: Path | None = None,
                    class_names=None) -> dict[str, Path | None]:
    """Write confusion-matrix PNG, classification-report TXT and HTML report.

    Returns paths keyed by 'cm_png', 'report_txt', 'report_html' (None if out_dir is None).
    """
    if out_dir is None:
        return {"cm_png": None, "report_txt": None, "report_html": None}
    out_dir.mkdir(parents=True, exist_ok=True)
    cm_png = out_dir / "confusion_matrix.png"
    report_txt = out_dir / "classification_report.txt"
    report_html = out_dir / "report.html"

    fig = confusion_matrix_plot(labels, preds, class_names=class_names, out_path=cm_png)
    text = classification_report_text(labels, preds, class_names=class_names)
    report_txt.write_text(text)

    metrics = compute_metrics(labels, preds, probs, class_names=class_names)
    report_html.write_text(html_report(metrics, fig, text))
    return {"cm_png": cm_png, "report_txt": report_txt, "report_html": report_html}
