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


def n_classes(labels, preds) -> int:
    labels, preds = _as_arrays(labels, preds)
    return int(max(labels.max(), preds.max())) + 1


def compute_metrics(labels, preds, probs=None, class_names=None) -> dict[str, float]:
    """Return a flat dict of scalar metrics ready for mlflow.log_metrics."""
    labels, preds = _as_arrays(labels, preds)
    n = n_classes(labels, preds)
    class_names = class_names or [f"KL {i}" for i in range(n)]

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "kappa": cohen_kappa_score(labels, preds, weights="quadratic"),
        "kappa_linear": cohen_kappa_score(labels, preds, weights="linear"),
    }

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

    return {k: float(v) for k, v in metrics.items()}


def classification_report_text(labels, preds, class_names=None, digits: int = 4) -> str:
    labels, preds = _as_arrays(labels, preds)
    n = n_classes(labels, preds)
    class_names = class_names or [f"KL {i}" for i in range(n)]
    return classification_report(labels, preds, target_names=class_names, digits=digits, zero_division=0)


def confusion_matrix_plot(labels, preds, class_names=None, out_path: Path | None = None) -> plt.Figure:
    labels, preds = _as_arrays(labels, preds)
    n = n_classes(labels, preds)
    class_names = class_names or [f"KL {i}" for i in range(n)]
    cm = confusion_matrix(labels, preds)
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
