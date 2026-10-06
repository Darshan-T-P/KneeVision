"""Generate all publication-ready figures for the KneeVision++ paper.

Generates high-resolution (300 DPI) publication figures from verified
experimental artifacts (models/*.json, reports/*.csv, reports/*.json):

  Fig 1: fig1_architecture_schematic.png - Ordinal Architecture & Soft MixUp
  Fig 2: fig2_ablation_comparison.png     - Ablation Comparison (QWK & MAE)
  Fig 3: fig3_test_confusion_matrix.png   - Final Test Confusion Matrix & Recall
  Fig 4: fig4_error_distribution.png      - Ordinal Error Distance Distribution
  Fig 5: fig5_per_class_metrics.png       - Per-Class Precision, Recall, & F1
  Fig 6: fig6_val_vs_test_generalization.png - Validation vs Test Generalization
  Fig 7: fig7_reliability_diagram.png     - Calibration Curves & Reliability Diagram

Usage:
    uv run python scripts/generate_publication_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Styling constants for publication
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cohesive clinical/research palette
PALETTE = {
    "primary": "#1f77b4",       # Deep blue
    "secondary": "#2ca02c",     # Forest green
    "accent": "#d62728",        # Coral red
    "purple": "#9467bd",        # Violet
    "orange": "#ff7f0e",        # Warm amber
    "cyan": "#17becf",          # Cyan
    "gray_dark": "#2c3e50",
    "gray_light": "#f8f9fa",
    "border": "#bdc3c7",
}


def plot_fig1_architecture_schematic():
    """Fig 1: Conceptual Architecture & Continuous Cumulative Target Formulation."""
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle="", color="#e8f4f8", border="#1f77b4", lw=1.5):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=color, edgecolor=border, linewidth=lw
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color="#1a252f")
            ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center",
                    fontsize=8.5, color="#555555", linespacing=1.2)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color="#1a252f")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#34495e", lw=1.8, shrinkA=3, shrinkB=3)
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label, ha="center", va="bottom",
                    fontsize=8.5, color="#2c3e50", fontweight="semibold")

    # Title (own band, clear of the "Continuous Cumulative Targets" box below)
    ax.text(6.0, 6.85, "DenseNet121 + CORAL Architecture with Soft Ordinal MixUp",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#1a252f")

    # Inputs & MixUp
    draw_box(0.5, 4.2, 2.2, 1.2, "Sample A (X_A, y_A)", "Grade y_A in {0..4}\nOne-hot vector", "#f4f6f7", "#7f8c8d")
    draw_box(0.5, 1.6, 2.2, 1.2, "Sample B (X_B, y_B)", "Grade y_B in {0..4}\nOne-hot vector", "#f4f6f7", "#7f8c8d")

    draw_arrow(2.7, 4.8, 3.4, 3.8)
    draw_arrow(2.7, 2.2, 3.4, 3.2)

    # Blending Box
    draw_box(3.4, 2.5, 2.3, 2.0, "MixUp Blending\n(alpha = 0.4)",
             "X_tilde = lambda*X_A + (1-lambda)*X_B\ny_tilde = lambda*y_A + (1-lambda)*y_B\nContinuous label simplex",
             "#fef9e7", "#f39c12")

    draw_arrow(5.7, 3.5, 6.3, 3.5)

    # Backbone
    draw_box(6.3, 2.5, 2.3, 2.0, "DenseNet121\nFeature Extractor",
             "Pretrained ImageNet weights\nGlobal Average Pooling\nOutput: 1024-d feature vector h",
             "#e8f8f5", "#16a085")

    draw_arrow(8.6, 3.5, 9.2, 3.5)

    # CORAL Ordinal Head
    draw_box(9.2, 2.3, 2.4, 2.4, "CORAL Ordinal Head\n(4 Binary Tasks)",
             "Task k in {1, 2, 3, 4}\ns_k = w^T h + b_k\nShared w, ordered b_k\nsigmoid(s_k) = P(y > k)",
             "#eaf2f8", "#2980b9")

    # Output Branches
    draw_arrow(10.4, 4.7, 10.4, 5.3)
    draw_box(8.8, 5.3, 3.1, 0.9, "Continuous Cumulative Targets",
             "t_k = sum_{c > k} y_tilde_c in [0, 1]\nLoss: sum_k BCEWithLogits(s_k, t_k)",
             "#fdf2e9", "#d35400")

    draw_arrow(10.4, 2.3, 10.4, 1.5)
    draw_box(8.8, 0.6, 3.1, 0.9, "Ordinal Rank Decoding",
             "y_hat = sum_{k=1}^4 I(sigmoid(s_k) >= 0.5)\nEnsures strict rank consistency",
             "#e8f8f5", "#27ae60")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "fig1_architecture_schematic.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig2_ablation_comparison():
    """Fig 2: Validation QWK and MAE Across Ablation Configurations (two single-axis panels)."""
    models = [
        "Baseline CORAL\n(Hard MixUp)",
        "Focal Loss\n(No MixUp)",
        "Focal Loss\n(MixUp a=0.2)",
        "Soft CORAL\n(No MixUp)",
        "Soft CORAL\n(MixUp a=0.2)",
        "Soft CORAL\n(MixUp a=0.4)\n[Champion]",
        "Soft CORAL\n(Seed 123)",
    ]
    qwks = [0.7785, 0.7831, 0.7753, 0.7722, 0.7971, 0.8009, 0.7945]
    maes = [0.4661, 0.4467, 0.4734, 0.4697, 0.4262, 0.4298, 0.4334]

    x = np.arange(len(models))
    width = 0.58

    # Two single-axis panels instead of a dual-axis (twinx) chart: QWK and MAE
    # have different scales and "better" directions, so they are never drawn
    # on a shared axis.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors_qwk = ["#7f8c8d", "#95a5a6", "#bdc3c7", "#3498db", "#2980b9", "#1b4f72", "#5499c7"]
    bars1 = ax1.bar(x, qwks, width, color=colors_qwk, edgecolor="#2c3e50", linewidth=0.8)
    ax1.set_ylabel("Validation QWK (higher is better)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0.70, 0.83)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_title("Quadratic Weighted Kappa", fontsize=11.5, fontweight="bold", pad=10)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.002, f"{h:.4f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1b4f72")

    colors_mae = ["#f5b7b1", "#f1948a", "#ec7063", "#e59866", "#d35400", "#922b21", "#ba4a00"]
    bars2 = ax2.bar(x, maes, width, color=colors_mae, edgecolor="#641e16", linewidth=0.8, hatch="//")
    ax2.set_ylabel("Validation MAE (lower is better)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0.38, 0.52)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.set_title("Mean Absolute Error", fontsize=11.5, fontweight="bold", pad=10)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.003, f"{h:.4f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#922b21")

    fig.suptitle("Ablation Study: Validation Performance Across Loss & MixUp Configurations",
                  fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = OUTPUT_DIR / "fig2_ablation_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig3_test_confusion_matrix():
    """Fig 3: Publication-Quality Confusion Matrix with Counts and Recall."""
    cm = np.array([
        [501, 120, 18, 0, 0],
        [99, 138, 59, 0, 0],
        [16, 135, 287, 9, 0],
        [0, 15, 106, 97, 5],
        [0, 0, 2, 15, 34],
    ])
    classes = ["KL0", "KL1", "KL2", "KL3", "KL4"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.17)
    cbar.set_label("Row-Normalized Recall", rotation=270, labelpad=15, fontweight="semibold")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=10.5, fontweight="semibold")
    ax.set_yticklabels(classes, fontsize=10.5, fontweight="semibold")

    ax.set_xlabel("Predicted Kellgren-Lawrence Grade", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Kellgren-Lawrence Grade", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Held-Out Test Confusion Matrix (N = 1,656 Radiographs)\nExact Accuracy: 63.83% | Within-1 Accuracy: 96.92%",
                 fontsize=11.5, fontweight="bold", pad=12)

    thresh = cm_norm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_color = "white" if cm_norm[i, j] > thresh else "#1a252f"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color=text_color, fontsize=9.5,
                    fontweight="bold" if i == j else "normal")

    # Annotate class totals on right (own margin, clear of the colorbar)
    row_sums = cm.sum(axis=1)
    for i, s in enumerate(row_sums):
        ax.text(4.62, i, f"n={s}", va="center", ha="left", fontsize=9, color="#555555",
                fontstyle="italic", clip_on=False)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "fig3_test_confusion_matrix.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig4_error_distribution():
    """Fig 4: Absolute Ordinal Error Distance Distribution."""
    distances = [0, 1, 2, 3, 4]
    counts = [1057, 548, 51, 0, 0]
    total = 1656
    percentages = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#27ae60", "#2980b9", "#e67e22", "#c0392b", "#7d3c98"]

    bars = ax.bar(distances, percentages, width=0.55, color=colors, edgecolor="#2c3e50", linewidth=0.9)

    ax.set_xlabel("Absolute Ordinal Error Distance (|y_true - y_pred|)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Percentage of Test Samples (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylim(0, 75)
    ax.set_xticks(distances)
    ax.set_xticklabels([
        "0\n(Exact Match)",
        "1\n(Adjacent Grade)",
        "2\n(Two Grades)",
        "3\n(Severe Error)",
        "4\n(Extreme Error)"
    ], fontsize=9.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, count, pct in zip(bars, counts, percentages):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, pct + 1.2,
                    f"{pct:.2f}%\n(n={count:,})",
                    ha="center", va="bottom", fontsize=9.2, fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 1.2,
                    "0.0%\n(n=0)",
                    ha="center", va="bottom", fontsize=9.2, color="#7f8c8d", fontstyle="italic")

    # Boundary callout
    ax.axvline(1.5, color="#c0392b", linestyle=":", linewidth=1.5)
    ax.text(1.55, 60, "Clinical Ordinal Safety Boundary:\nWithin-1 Accuracy: 96.92%\nWithin-2 Accuracy: 100.00%\nZero Extreme Errors (>= 3)",
            fontsize=9.5, color="#922b21", fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fadbd8", edgecolor="#e6b0aa", alpha=0.9))

    ax.set_title("Ordinal Error Distribution on Held-Out Test Set (N = 1,656)",
                 fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "fig4_error_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig5_per_class_metrics():
    """Fig 5: Per-Class Precision, Recall, and F1 with Discordance Annotations."""
    classes = ["KL0 (None)", "KL1 (Doubtful)", "KL2 (Mild)", "KL3 (Moderate)", "KL4 (Severe)"]
    precision = [0.8133, 0.3382, 0.6081, 0.8017, 0.8718]
    recall = [0.7840, 0.4662, 0.6421, 0.4350, 0.6667]
    f1 = [0.7984, 0.3920, 0.6246, 0.5640, 0.7556]
    supports = [639, 296, 447, 223, 51]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    b1 = ax.bar(x - width, precision, width, label="Precision", color="#3498db", edgecolor="#1b4f72", linewidth=0.8)
    b2 = ax.bar(x, recall, width, label="Recall", color="#2ecc71", edgecolor="#145a32", linewidth=0.8)
    b3 = ax.bar(x + width, f1, width, label="F1-Score", color="#9b59b6", edgecolor="#4a235a", linewidth=0.8)

    ax.set_ylabel("Score", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(classes, supports)], fontsize=9.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    ax.set_title("Per-Class Test Performance (Precision, Recall, F1) across KL Grades",
                 fontsize=12, fontweight="bold", pad=12)

    # Highlight KL1 discordance and KL3 under-grading
    ax.annotate("High Inter-Rater\nDiscordance\n(KL1 Recall 46.6%)",
                xy=(1, 0.47), xytext=(1.0, 0.82),
                arrowprops=dict(facecolor="#e67e22", shrink=0.08, width=1.2, headwidth=6),
                ha="center", fontsize=8.5, fontweight="bold", color="#b9770e",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef9e7", edgecolor="#f5b041", alpha=0.9))

    ax.annotate("Conservative Under-Grading\n(47.5% KL3 pred as KL2)",
                xy=(3, 0.44), xytext=(3.0, 0.78),
                arrowprops=dict(facecolor="#c0392b", shrink=0.08, width=1.2, headwidth=6),
                ha="center", fontsize=8.5, fontweight="bold", color="#922b21",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", edgecolor="#e6b0aa", alpha=0.9))

    fig.tight_layout()
    out_path = OUTPUT_DIR / "fig5_per_class_metrics.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig6_val_vs_test_generalization():
    """Fig 6: Validation vs Held-Out Test Generalization Comparison with 95% CIs."""
    metrics = ["QWK", "Exact Acc (%)", "Within-1 (%)", "Macro-F1", "MAE"]
    val_scores = [0.8009, 61.02, 96.13, 0.6271, 0.4298]
    test_scores = [0.8232, 63.83, 96.92, 0.6269, 0.3925]

    # Empirical 95% cluster bootstrap CIs from reports/final_ordinal_soft_mixup_a04_test.predictions_bootstrap.json
    ci_lower = [0.8036, 61.23, 96.14, 0.6050, 0.3641]
    ci_upper = [0.8398, 66.31, 97.77, 0.6480, 0.4215]
    yerr_lower = [test_scores[i] - ci_lower[i] for i in range(len(metrics))]
    yerr_upper = [ci_upper[i] - test_scores[i] for i in range(len(metrics))]
    yerr = [yerr_lower, yerr_upper]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), gridspec_kw={"width_ratios": [3, 2]})

    # Normalized / Kappa metrics
    x1 = np.arange(3)
    width = 0.35
    ax_left = axes[0]
    ax_left.bar(x1 - width / 2, [val_scores[0], val_scores[3], val_scores[4]], width,
                label="Validation Split", color="#7f8c8d", edgecolor="#2c3e50")
    ax_left.errorbar(x1 + width / 2, [test_scores[0], test_scores[3], test_scores[4]],
                     yerr=[[yerr_lower[0], yerr_lower[3], yerr_lower[4]],
                           [yerr_upper[0], yerr_upper[3], yerr_upper[4]]],
                     fmt="none", ecolor="#1b4f72", elinewidth=1.5, capsize=4)
    ax_left.bar(x1 + width / 2, [test_scores[0], test_scores[3], test_scores[4]], width,
                label="Held-Out Test (95% Clustered CI)", color="#2980b9", edgecolor="#1b4f72")

    ax_left.set_xticks(x1)
    ax_left.set_xticklabels(["QWK (Quadratic Kappa)", "Macro-F1", "MAE (Lower is better)"], fontsize=9.5)
    ax_left.set_ylabel("Metric Value", fontsize=10.5, fontweight="bold")
    ax_left.set_ylim(0, 1.0)
    ax_left.grid(axis="y", linestyle="--", alpha=0.5)
    ax_left.legend(loc="upper right", framealpha=0.9)
    ax_left.set_title("Agreement, Distance, & Balance Metrics", fontsize=11, fontweight="bold")

    # Percentage metrics
    x2 = np.arange(2)
    ax_right = axes[1]
    ax_right.bar(x2 - width / 2, [val_scores[1], val_scores[2]], width,
                 label="Validation", color="#7f8c8d", edgecolor="#2c3e50")
    ax_right.errorbar(x2 + width / 2, [test_scores[1], test_scores[2]],
                      yerr=[[yerr_lower[1], yerr_lower[2]],
                            [yerr_upper[1], yerr_upper[2]]],
                      fmt="none", ecolor="#145a32", elinewidth=1.5, capsize=4)
    ax_right.bar(x2 + width / 2, [test_scores[1], test_scores[2]], width,
                 label="Held-Out Test", color="#27ae60", edgecolor="#145a32")

    ax_right.set_xticks(x2)
    ax_right.set_xticklabels(["Exact Accuracy (%)", "Within-1 Accuracy (%)"], fontsize=9.5)
    ax_right.set_ylabel("Percentage (%)", fontsize=10.5, fontweight="bold")
    ax_right.set_ylim(50, 102)
    ax_right.grid(axis="y", linestyle="--", alpha=0.5)
    ax_right.legend(loc="lower right", framealpha=0.9)
    ax_right.set_title("Accuracy & Clinical Tolerance", fontsize=11, fontweight="bold")

    fig.suptitle("Validation vs Held-Out Test Generalization (Locked Model: Epoch 25, Seed 42)",
                 fontsize=12.5, fontweight="bold", y=1.01)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "fig6_val_vs_test_generalization.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig7_reliability_diagram():
    """Fig 7: Calibration Curves & Reliability Diagram with ECE Callout."""
    csv_path = PROJECT_ROOT / "models" / "final_ordinal_soft_mixup_a04_test.predictions.csv"
    if not csv_path.exists():
        print(f"Skipping Fig 7: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    labels = df["true_label"].to_numpy()
    prob_cols = [f"prob_KL{c}" for c in range(5)]
    probs = df[prob_cols].to_numpy()
    n_samples, n_classes = probs.shape

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # Reference diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration (Ideal)")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for c in range(n_classes):
        c_probs = probs[:, c]
        c_labels = (labels == c).astype(float)
        bin_accs, bin_confs = [], []

        for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            mask = (c_probs >= lo) & (c_probs <= hi) if i == len(bin_edges) - 2 else (c_probs >= lo) & (c_probs < hi)
            if mask.sum() > 0:
                bin_accs.append(c_labels[mask].mean())
                bin_confs.append(c_probs[mask].mean())

        c_brier = np.mean((c_probs - c_labels) ** 2)
        ax.plot(bin_confs, bin_accs, marker="o", lw=1.8, markersize=5,
                color=colors[c], label=f"KL{c} (OvR Brier: {c_brier:.3f})")

    ax.set_xlabel("Mean Predicted Probability (Confidence)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Empirical True Fraction (Accuracy)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Calibration annotation
    ax.text(0.04, 0.88,
            "Calibration Metrics (Closed Bin Boundary):\n"
            "- Macro ECE: 0.0496 (4.96%)\n"
            "- Macro Brier Score: 0.0947\n"
            "- Multiclass Brier Score: 0.4736",
            fontsize=9.5, fontweight="semibold", color="#1b4f72",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#eaf2f8", edgecolor="#aed6f1", alpha=0.9))

    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.set_title("Reliability Diagram (Calibration Curves) on Held-Out Test Set\n(N = 1,656 Samples, 10 Bins)",
                 fontsize=11.5, fontweight="bold", pad=12)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "fig7_reliability_diagram.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    print(f"Generating publication figures to {OUTPUT_DIR}...")
    plot_fig1_architecture_schematic()
    plot_fig2_ablation_comparison()
    plot_fig3_test_confusion_matrix()
    plot_fig4_error_distribution()
    plot_fig5_per_class_metrics()
    plot_fig6_val_vs_test_generalization()
    plot_fig7_reliability_diagram()
    print("All publication figures successfully generated!")


if __name__ == "__main__":
    main()
