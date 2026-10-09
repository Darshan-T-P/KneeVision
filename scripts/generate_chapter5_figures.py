"""Generate the figures requested for Chapter 5 from saved KneeVision results.

Run from the repository root with:
    .venv/bin/python scripts/generate_chapter5_figures.py

Use --skip-xai to regenerate only the metric figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from PIL import Image
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "chapter5_figures"
CLASS_NAMES = ["KL 0", "KL 1", "KL 2", "KL 3", "KL 4"]
COLORS = ["#286f8e", "#65a765", "#e2b93b", "#e77c42", "#a94442"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    labels = np.array([int(row["true_label"]) for row in rows])
    probs = np.array([
        [float(row[f"prob_KL{grade}"]) for grade in range(5)] for row in rows
    ])
    return labels, probs, rows


def save_figure(fig: plt.Figure, name: str) -> None:
    path = OUTPUT / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"Saved {path.relative_to(ROOT)}")


def plot_baseline_performance() -> None:
    metrics = read_json(REPORTS / "test_evals" / "best_densenet121.test.json")
    names = ["Accuracy", "QWK", "Macro F1"]
    values = [metrics["accuracy"], metrics["qwk"], metrics["macro_f1"]]
    colors = ["#286f8e", "#65a765", "#e2b93b"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(names, values, color=colors, width=0.58)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Score")
    ax.set_title("Image-only baseline performance on the held-out test set")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025,
                f"{value * 100:.1f}%" if value == values[0] else f"{value:.3f}",
                ha="center", va="bottom", fontweight="bold")
    save_figure(fig, "fig_5_1_image_only_baseline.png")


def plot_per_class_metrics() -> None:
    metrics = read_json(MODELS / "final_ordinal_soft_mixup_a04_test.json")
    rows = metrics["per_class"]
    metric_names = ["precision", "recall", "f1"]
    labels = ["Precision", "Recall", "F1"]
    x = np.arange(5)
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for offset, (key, label, color) in enumerate(zip(metric_names, labels, COLORS[:3])):
        values = [row[key] for row in rows]
        ax.bar(x + (offset - 1) * width, values, width, label=label, color=color)
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Per-class test performance of the final ordinal model")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.22)
    save_figure(fig, "fig_5_2_per_class_precision_recall_f1.png")


def copy_multimodal_confusion_matrix() -> None:
    source = REPORTS / "evaluate_fusion" / "confusion_matrix.png"
    if not source.is_file():
        raise FileNotFoundError(f"Expected saved multimodal confusion matrix: {source}")
    destination = OUTPUT / "fig_5_3_multimodal_confusion_matrix.png"
    shutil.copyfile(source, destination)
    print(f"Copied {destination.relative_to(ROOT)}")


def plot_roc_pr(labels: np.ndarray, probs: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for grade, color in enumerate(COLORS):
        binary_labels = labels == grade
        fpr, tpr, _ = roc_curve(binary_labels, probs[:, grade])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{CLASS_NAMES[grade]} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1, label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate",
           ylabel="True positive rate", title="One-vs-rest ROC curves on the held-out test set")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.2)
    save_figure(fig, "fig_5_4_one_vs_rest_roc.png")

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for grade, color in enumerate(COLORS):
        precision, recall, _ = precision_recall_curve(labels == grade, probs[:, grade])
        average_precision = average_precision_score(labels == grade, probs[:, grade])
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{CLASS_NAMES[grade]} (AP {average_precision:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
           title="One-vs-rest precision-recall curves on the held-out test set")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax.grid(alpha=0.2)
    save_figure(fig, "fig_5_5_precision_recall.png")


def plot_training_curves() -> None:
    with sqlite3.connect(ROOT / "mlflow.db") as connection:
        run = connection.execute(
            """SELECT runs.run_uuid
               FROM runs
               JOIN tags ON tags.run_uuid = runs.run_uuid
               WHERE runs.experiment_id = 1
                 AND tags.key = 'mlflow.runName'
                 AND tags.value LIKE 'densenet121_ordinal_ordinal_soft_mixup_a04_%'
                 AND tags.value NOT LIKE '%seed123%'
               ORDER BY runs.start_time DESC
               LIMIT 1"""
        ).fetchone()
        if run is None:
            raise ValueError("Could not find the final alpha=0.4 ordinal run in mlflow.db")
        history = connection.execute(
            """SELECT key, value, step FROM metrics
               WHERE run_uuid = ? AND key IN ('train_loss', 'val_loss', 'val_kappa')
               ORDER BY step""",
            (run[0],),
        ).fetchall()

    history_by_epoch: dict[int, dict[str, float]] = {}
    for key, value, step in history:
        history_by_epoch.setdefault(step, {})[key] = value
    records = [
        (epoch, values["train_loss"], values["val_loss"], values["val_kappa"])
        for epoch, values in sorted(history_by_epoch.items())
        if all(key in values for key in ("train_loss", "val_loss", "val_kappa"))
    ]
    if not records:
        raise ValueError("Final ordinal MLflow run has no complete per-epoch metrics")

    epochs, train_loss, val_loss, qwk = np.array(records).T
    best_epoch = int(epochs[np.argmax(qwk)])
    fig, (loss_ax, qwk_ax) = plt.subplots(1, 2, figsize=(10, 4.1))
    loss_ax.plot(epochs, train_loss, "o-", ms=3, lw=1.8, color="#286f8e", label="Training loss")
    loss_ax.plot(epochs, val_loss, "s--", ms=3, lw=1.8, color="#a94442", label="Validation loss")
    loss_ax.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    loss_ax.legend(frameon=False)
    loss_ax.grid(alpha=0.2)
    qwk_ax.plot(epochs, qwk, "o-", ms=3, lw=1.8, color="#468a62", label="Validation QWK")
    qwk_ax.axhline(0.8, color="#b27b25", ls=":", lw=1.4, label="QWK = 0.80")
    qwk_ax.axvline(best_epoch, color="#555555", ls="--", lw=1, alpha=0.7,
                   label=f"Best epoch ({best_epoch})")
    qwk_ax.set(xlabel="Epoch", ylabel="Quadratic weighted kappa", ylim=(0, 1), title="Validation agreement")
    qwk_ax.legend(frameon=False)
    qwk_ax.grid(alpha=0.2)
    fig.suptitle("Final DenseNet121 ordinal Soft MixUp run (α = 0.4, seed 42)", fontsize=11)
    save_figure(fig, "fig_5_6_training_validation_learning_curves.png")


def plot_ordinal_error(labels: np.ndarray, probs: np.ndarray) -> None:
    errors = np.abs(probs.argmax(axis=1) - labels)
    percentages = np.array([(errors == distance).mean() * 100 for distance in range(5)])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(np.arange(5), percentages, color=COLORS, width=0.62)
    ax.set_xticks(np.arange(5), [f"Δ = {distance}" for distance in range(5)])
    ax.set_ylabel("Test cases (%)")
    ax.set_ylim(0, max(percentages) * 1.18)
    ax.set_title("Absolute ordinal error distribution")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(percentages) * 0.025,
                f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    save_figure(fig, "fig_5_7_ordinal_error_distribution.png")


def plot_modality_comparison() -> None:
    metrics = read_json(REPORTS / "evaluate_fusion" / "metrics.json")
    modalities = ["Image (best_densenet121)", "Clinical Text (BioClinicalBERT)", "Multimodal Fusion"]
    display = ["Image only", "Structured clinical text", "Image + text"]
    series = [("accuracy", "Accuracy", "#286f8e"),
              ("kappa", "QWK", "#65a765"),
              ("macro_f1", "Macro F1", "#e2b93b")]
    x = np.arange(len(display))
    width = 0.23
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, (key, label, color) in enumerate(series):
        values = [metrics[name][key] for name in modalities]
        ax.bar(x + (offset - 1) * width, values, width, label=label, color=color)
        for bar, value in zip(ax.containers[-1], values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=0)
    ax.set_xticks(x, display)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Modality comparison (5-grade test evaluation, n = 1,656)")
    ax.text(0.5, -0.19,
            "Clinical text uses structured OARSI radiographic fields, not free-form clinical notes.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#555555")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "fig_5_8_modality_comparison.png")


def plot_internal_qwk_comparison() -> None:
    entries = [
        ("Image-only\nbaseline", "best_densenet121.test.json"),
        ("CORAL\nhard targets", None),
        ("Focal loss\n(no MixUp)", "best_densenet121_focal_nomixup.test.json"),
        ("Soft CORAL\nMixUp α=0.2", "best_densenet121_ordinal_ordinal_soft_mixup.test.json"),
        ("Soft CORAL\nMixUp α=0.4", "final_ordinal_soft_mixup_a04_test.json"),
    ]
    values = []
    for _, filename in entries:
        if filename is None:
            values.append(0.779781)
        else:
            path = (MODELS / filename if filename.startswith("final_")
                    else REPORTS / "test_evals" / filename)
            values.append(read_json(path)["qwk"])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#9ba6aa", "#9ba6aa", "#7f9d8a", "#7f9d8a", "#286f8e"]
    bars = ax.bar(np.arange(len(entries)), values, color=colors, width=0.62)
    ax.set_xticks(np.arange(len(entries)), [label for label, _ in entries])
    ax.set_ylim(0.72, 0.86)
    ax.set_ylabel("Quadratic weighted kappa (QWK)")
    ax.set_title("Held-out test QWK across evaluated KneeVision approaches")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.text(0.5, -0.21,
            "Internal model comparison only; these are not published external-study benchmarks.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#555555")
    save_figure(fig, "fig_5_9_internal_approach_qwk_comparison.png")


def blend_overlay(image: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    source = np.asarray(image.resize((224, 224)).convert("RGB"), dtype=np.float32) / 255
    heatmap = np.asarray(heatmap, dtype=np.float32)
    cmap = matplotlib.colormaps["turbo"]
    colored = cmap(Normalize(vmin=float(heatmap.min()), vmax=float(heatmap.max()) + 1e-8)(heatmap))[..., :3]
    return np.clip(source * 0.52 + colored * 0.48, 0, 1)


def plot_xai_panel(rows: list[dict[str, str]], labels: np.ndarray, probs: np.ndarray) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from kneevision.data.transforms import val_transform
    from kneevision.models.image_model import load_trained_model
    from kneevision.xai.gradcam import GradCAM
    from kneevision.xai.lime import explain as explain_lime
    from kneevision.xai.scorecam import ScoreCAM

    candidates = [
        index for index, row in enumerate(rows)
        if labels[index] == 3 and int(row["pred_label"]) == 3
    ]
    if not candidates:
        raise ValueError("No correctly classified KL 3 radiograph is available for the XAI panel")
    index = max(candidates, key=lambda item: probs[item, 3])
    np.random.seed(42)
    image = Image.open(rows[index]["image_path"]).convert("RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(
        MODELS / "best_densenet121_ordinal_ordinal_soft_mixup_a04.pt", device, num_classes=5
    )
    tensor = val_transform(image).unsqueeze(0).to(device)
    target_grade = int(rows[index]["pred_label"])

    print(f"Generating XAI maps on {device}; Score-CAM evaluates the full feature map.")
    gradcam_map = GradCAM(model).generate(tensor, class_idx=target_grade)
    scorecam_map = ScoreCAM(model).generate(tensor, class_idx=target_grade)
    _, _, lime_map, _ = explain_lime(
        model, image, val_transform, device, grid_size=7, num_samples=250
    )

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.7))
    panels = [
        (np.asarray(image.resize((224, 224)).convert("RGB")), "Radiograph"),
        (blend_overlay(image, gradcam_map), "Grad-CAM"),
        (blend_overlay(image, scorecam_map), "Score-CAM"),
        (blend_overlay(image, lime_map), "LIME (superpixel grid)"),
    ]
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f"Explanations for one KL {labels[index]} test radiograph | predicted KL {target_grade}",
        fontsize=11,
    )
    save_figure(fig, "fig_5_10_gradcam_scorecam_lime.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-xai", action="store_true", help="Generate figures except the CAM/LIME panel")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    labels, probs, rows = read_predictions(MODELS / "final_ordinal_soft_mixup_a04_test.predictions.csv")
    plot_baseline_performance()
    plot_per_class_metrics()
    copy_multimodal_confusion_matrix()
    plot_roc_pr(labels, probs)
    plot_training_curves()
    plot_ordinal_error(labels, probs)
    plot_modality_comparison()
    plot_internal_qwk_comparison()
    if not args.skip_xai:
        plot_xai_panel(rows, labels, probs)
    print(f"Figure output directory: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()