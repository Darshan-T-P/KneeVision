"""Clean slide figures (minimal text, no baked-in footnotes) into slide_figures_clean/.

All values are read from result files; nothing is typed by hand except the base paper's published binary accuracy.
Usage: uv run python scripts/make_clean_figures.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "slide_figures_clean"
OUT.mkdir(exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
T = ROOT / "reports" / "test_evals"
MT = "best_densenet121_multitask_ordinal_soft_mixup_a04"
BR = json.loads((ROOT / "reports/branches_tasks/branch_task_results.json").read_text())
ROWS = json.loads((ROOT / "reports/paper_metrics.json").read_text())


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def tj(stem):
    return json.loads((T / f"{stem}.test.json").read_text())


def load_preds(path):
    y, P = [], []
    for r in csv.DictReader(open(path)):
        y.append(int(r["true_label"])); P.append([float(r[f"prob_KL{k}"]) for k in range(5)])
    return np.array(y), np.array(P)


def style(ax, ylabel=None):
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


# 1. accuracy by task and branch
mrows = [r for r in ROWS if "aux heads" in r["model"] and "40-epoch" not in r["model"]]
xr = [np.mean([r[k] for r in mrows]) * 100 for k in ("bin_acc", "tri_acc", "acc")]
cl = [BR["clinical"][k] * 100 for k in ("acc_binary", "acc_3class_01_2_34", "acc5")]
fu = [BR["fusion"][k] * 100 for k in ("acc_binary", "acc_3class_01_2_34", "acc5")]
fig, ax = plt.subplots(figsize=(10, 5.2))
x = np.arange(3); w = 0.27
for i, (lab, v, c) in enumerate((("X-ray only", xr, BLUE), ("Clinical", cl, ORANGE), ("Fusion", fu, AQUA))):
    bars = ax.bar(x + (i - 1) * w, v, w * 0.92, color=c, label=lab)
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, val + 1.2, f"{val:.1f}", ha="center", fontsize=13, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["2 classes", "3 classes", "5 classes"], fontsize=14)
ax.set_ylim(0, 108); style(ax, "Accuracy (%)"); ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=13)
save(fig, "1_accuracy_by_task.png")

# 2. confusion matrix, X-ray (<1% shown for tiny non-zero cells)
cm = np.array(tj(MT)["confusion_matrix"]); pct = cm / cm.sum(1, keepdims=True) * 100
fig, ax = plt.subplots(figsize=(6.4, 5.6))
ax.imshow(pct, cmap=LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#2a78d6", "#10335f"]), vmin=0, vmax=100)
for i in range(5):
    for j in range(5):
        p = f"{pct[i, j]:.0f}%" if pct[i, j] >= 1 or cm[i, j] == 0 else "<1%"
        ax.text(j, i, f"{cm[i, j]}\n{p}", ha="center", va="center", fontsize=13, fontweight="bold" if i == j else "normal",
                color="white" if pct[i, j] > 55 else INK)
labs = [f"KL{k}" for k in range(5)]
ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(labs); ax.set_yticklabels(labs)
ax.set_xlabel("Predicted grade"); ax.set_ylabel("True grade"); ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
save(fig, "2_confusion_matrix.png")

# 3. ROC, binary task
y5, P5 = load_preds(T / f"{MT}.predictions.csv")
yf, Pf = load_preds(ROOT / "reports/branches_tasks/fusion_predictions.csv")
fig, ax = plt.subplots(figsize=(5.8, 5.4))
for lab, y, P, c in (("X-ray only", y5, P5, BLUE), ("Fusion", yf, Pf, AQUA)):
    yb, s = (y >= 2).astype(int), P[:, 2:].sum(1)
    fpr, tpr, _ = roc_curve(yb, s)
    ax.plot(fpr, tpr, color=c, lw=2.6, label=f"{lab} (AUC {roc_auc_score(yb, s):.2f})")
ax.plot([0, 1], [0, 1], color=GRAY, lw=1.2, ls="--")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.grid(True, color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right", fontsize=13)
save(fig, "3_roc_binary.png")

# 4. how far off are the predictions
d = {k: sum(cm[i, j] for i in range(5) for j in range(5) if abs(i - j) == k) for k in range(4)}
d[3] = sum(cm[i, j] for i in range(5) for j in range(5) if abs(i - j) >= 3)
vals = [d[0], d[1], d[2], d[3]]; tot = cm.sum()
fig, ax = plt.subplots(figsize=(7.4, 4.8))
bars = ax.bar(["Exact", "1 grade off", "2 grades off", "3+ off"], [v / tot * 100 for v in vals], color=[BLUE, BLUE, BLUE, BLUE], width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v / tot * 100 + 1.2, f"{v / tot * 100:.1f}%", ha="center", fontsize=14, fontweight="bold")
ax.set_ylim(0, 80); style(ax, "Share of test images (%)")
save(fig, "4_error_distribution.png")

# 5. model comparison
G = [("CORAL + OARSI heads", lambda m: "aux heads" in m and "40-epoch" not in m, True), ("Swin-T", lambda m: m.startswith("Swin"), False),
     ("CORAL DenseNet121", lambda m: m.startswith("DenseNet121 · soft CORAL + MixUp 0.4 · seed"), False),
     ("ConvNeXt-Tiny", lambda m: m.startswith("ConvNeXt"), False), ("Focal DenseNet121", lambda m: "Focal, no MixUp" in m, False),
     ("EfficientNet-B4", lambda m: m.startswith("EfficientNet"), False)]
D = sorted([(n, np.mean([r["qwk"] for r in ROWS if f(r["model"])]), h) for n, f, h in G], key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(9, 4.8))
for i, (n, m, h) in enumerate(D):
    ax.barh(i, m, color=BLUE if h else GRAY, height=0.62)
    ax.text(m + 0.003, i, f"{m:.3f}", va="center", fontsize=13, fontweight="bold" if h else "normal")
ax.set_yticks(range(len(D))); ax.set_yticklabels([d_[0] for d_ in D], fontsize=13); ax.set_xlim(0.70, 0.86)
ax.set_xlabel("Quadratic weighted kappa"); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
save(fig, "5_model_comparison.png")

# 6. binary accuracy vs base paper
fig, ax = plt.subplots(figsize=(8.4, 4.9))
v = [86.29, np.mean([r["bin_acc"] for r in mrows]) * 100, BR["fusion"]["acc_binary"] * 100]
bars = ax.bar(["Base paper", "Ours\n(X-ray only)", "Ours\n(fusion, uses\nclinical findings)"], v, color=[GRAY, BLUE, AQUA], width=0.55)
bars[2].set_hatch("//"); bars[2].set_edgecolor("white")
for b, val in zip(bars, v):
    ax.text(b.get_x() + b.get_width() / 2, val + 1.2, f"{val:.1f}", ha="center", fontsize=15, fontweight="bold")
ax.set_ylim(0, 108); style(ax, "Binary accuracy (%)")
save(fig, "6_binary_vs_base_paper.png")

# 7. example Grad-CAM heatmaps (final model, correctly graded test images)
import torch
from PIL import Image

from kneevision.data.transforms import val_transform
from kneevision.models.image_model import load_trained_model
from kneevision.utils.helpers import get_device
from kneevision.xai.gradcam import explain

dev = get_device()
model = load_trained_model(ROOT / f"models/{MT}.pt", dev, num_classes=5).to(dev).eval()
rows_ = list(csv.DictReader(open(T / f"{MT}.predictions.csv")))
pick = {}
for r in rows_:
    t = int(r["true_label"])
    if t in (0, 2, 4) and int(r["pred_label"]) == t and t not in pick:
        pick[t] = r["image_path"]
fig, axs = plt.subplots(2, 3, figsize=(9.6, 6.6))
for c, (g, pth) in enumerate(sorted(pick.items())):
    img = Image.open(pth).convert("RGB")
    _, _, overlay = explain(model, img, val_transform, dev)
    axs[0, c].imshow(img.resize((224, 224)), cmap="gray"); axs[1, c].imshow(overlay)
    axs[0, c].set_title(f"KL{g}", fontsize=16, fontweight="bold")
    for r_ in (0, 1):
        axs[r_, c].axis("off")
axs[0, 0].text(-0.06, 0.5, "X-ray", transform=axs[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=14, color=MUTED)
axs[1, 0].text(-0.06, 0.5, "Grad-CAM", transform=axs[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=14, color=MUTED)
fig.tight_layout()
save(fig, "7_gradcam_examples.png")
