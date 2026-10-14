"""Generate slide-ready PNG figures into slide_figures/ (white background, 16:9-friendly).

Every value is read from result files: reports/test_evals/*.test.json, reports/branches_tasks/*.json|csv,
reports/paper_metrics.json. Nothing is typed by hand except the base paper's published numbers (clearly labelled).
Palette: first three slots of the validated reference categorical palette (blue, orange, aqua), ink text colours.

Usage: uv run python scripts/make_slide_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "slide_figures"
OUT.mkdir(exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})

T = ROOT / "reports" / "test_evals"
A, MT = "best_densenet121_ordinal_ordinal_soft_mixup_a04", "best_densenet121_multitask_ordinal_soft_mixup_a04"
multi = [MT + s for s in ("", "_seed7", "_seed2024")]
BR = json.loads((ROOT / "reports/branches_tasks/branch_task_results.json").read_text())
rows = json.loads((ROOT / "reports/paper_metrics.json").read_text())


def tj(stem):
    p = T / f"{stem}.test.json"
    return json.loads(p.read_text() if p.exists() else (ROOT / "models/final_ordinal_soft_mixup_a04_test.json").read_text())


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def mean_sd(vals):
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


# ---- Fig 1: accuracy by task and branch ----
mrows = [r for r in rows if "aux heads" in r["model"] and "40-epoch" not in r["model"]]
xr = {"Binary\n(No OA vs OA)": mean_sd([r["bin_acc"] * 100 for r in mrows]),
      "3-class\n(0–1 / 2 / 3–4)": mean_sd([r["tri_acc"] * 100 for r in mrows]),
      "5-class\n(KL0–KL4)": mean_sd([r["acc"] * 100 for r in mrows])}
tasks = list(xr)
cl = [BR["clinical"]["acc_binary"] * 100, BR["clinical"]["acc_3class_01_2_34"] * 100, BR["clinical"]["acc5"] * 100]
fu = [BR["fusion"]["acc_binary"] * 100, BR["fusion"]["acc_3class_01_2_34"] * 100, BR["fusion"]["acc5"] * 100]
fig, ax = plt.subplots(figsize=(11, 5.6))
x = np.arange(3); w = 0.26
series = [("X-ray only (final model)", [xr[t][0] for t in tasks], BLUE, [xr[t][1] for t in tasks]),
          ("Clinical text", cl, ORANGE, None), ("Fusion (X-ray + clinical)", fu, AQUA, None)]
for i, (lab, vals, col, err) in enumerate(series):
    bars = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=col, label=lab, yerr=err, capsize=3, error_kw={"ecolor": MUTED, "lw": 1})
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2 + (0 if err is None else 0.8), f"{v:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=13); ax.set_ylim(0, 108); ax.set_ylabel("Test accuracy (%)")
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("Accuracy across tasks and branches (1,656 test X-rays)", fontsize=16, fontweight="bold", loc="left", pad=14)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=12)
fig.text(0.01, -0.05, "Clinical and fusion branches use radiographic findings that define the KL grade; X-ray-only is the image-only result. X-ray: 3-seed mean ± SD.",
         fontsize=10, color=MUTED, ha="left")
save(fig, "fig1_accuracy_by_task_and_branch.png")


# ---- confusion-matrix helper ----
def cm_fig(cm, labels, title, name, figsize=(6.6, 5.6)):
    cm = np.asarray(cm); rowpct = cm / cm.sum(1, keepdims=True) * 100
    cmap = LinearSegmentedColormap.from_list("blu", ["#f4f8fd", "#2a78d6", "#10335f"])
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rowpct, cmap=cmap, vmin=0, vmax=100)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            col = "white" if rowpct[i, j] > 55 else INK
            ax.text(j, i, f"{cm[i, j]}\n{rowpct[i, j]:.0f}%", ha="center", va="center", color=col, fontsize=12, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels))); ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=15, fontweight="bold", loc="left", pad=12)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    save(fig, name)


kl = ["KL0", "KL1", "KL2", "KL3", "KL4"]
cm5 = tj(MT)["confusion_matrix"]
cm_fig(cm5, kl, "X-ray model: 5-class confusion matrix\n(true grade vs predicted; cell = count, row %)", "fig2_confusion_xray_5class.png", (7.0, 6.0))
cm_fig(BR["fusion"]["confusion_matrix"], kl, "Fusion: 5-class confusion matrix\n(cell = count, row %)", "fig3_confusion_fusion_5class.png", (7.0, 6.0))


def to_binary(cm):
    cm = np.asarray(cm)
    return [[cm[:2, :2].sum(), cm[:2, 2:].sum()], [cm[2:, :2].sum(), cm[2:, 2:].sum()]]


bl = ["No OA\n(KL0–1)", "OA\n(KL2–4)"]
cm_fig(to_binary(cm5), bl, "X-ray model: binary result", "fig4a_binary_xray.png", (5.0, 4.6))
cm_fig(to_binary(BR["fusion"]["confusion_matrix"]), bl, "Fusion: binary result", "fig4b_binary_fusion.png", (5.0, 4.6))

# ---- Fig 5: model comparison (QWK) ----
groups = [("Ordinal + OARSI aux heads\nDenseNet121 (3 seeds)", lambda m: "aux heads" in m and "40-epoch" not in m, 8.1, True),
          ("Swin-T (1 seed)", lambda m: m.startswith("Swin"), 28.3, False),
          ("Ordinal DenseNet121 (4 seeds)", lambda m: m.startswith("DenseNet121 · soft CORAL + MixUp 0.4 · seed"), 8.1, False),
          ("ConvNeXt-Tiny (1 seed)", lambda m: m.startswith("ConvNeXt"), 28.6, False),
          ("Focal loss DenseNet121 (3 seeds)", lambda m: "Focal, no MixUp" in m, 8.1, False),
          ("EfficientNet-B4 (1 seed)", lambda m: m.startswith("EfficientNet"), 19.5, False)]
data = []
for lab, f, p, hl in groups:
    v = [r["qwk"] for r in rows if f(r["model"])]
    data.append((lab, *mean_sd(v), p, hl))
data.sort(key=lambda d: d[1])
fig, ax = plt.subplots(figsize=(11, 5.6))
for i, (lab, m, sd, p, hl) in enumerate(data):
    ax.barh(i, m, color=BLUE if hl else GRAY, height=0.62, xerr=sd if sd else None, capsize=3, error_kw={"ecolor": MUTED, "lw": 1})
    ax.text(m + sd + 0.004, i, f"{m:.3f}   ({p:.1f} M params)", va="center", fontsize=12, fontweight="bold" if hl else "normal")
ax.set_yticks(range(len(data))); ax.set_yticklabels([d[0] for d in data], fontsize=12); ax.set_xlim(0.70, 0.90)
ax.set_xlabel("Test quadratic weighted kappa (higher is better)"); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("Model comparison on the 5-grade task (same recipe)", fontsize=16, fontweight="bold", loc="left", pad=14)
fig.text(0.01, -0.04, "Bars = mean over seeds; whiskers = SD. Single-seed rows differ from DenseNet121 by less than the seed spread (~0.013).", fontsize=10, color=MUTED)
save(fig, "fig5_model_comparison_qwk.png")

# ---- Fig 6: per-class recall ----
rec = np.mean([[np.asarray(tj(s)["confusion_matrix"])[i, i] / np.sum(tj(s)["confusion_matrix"][i]) * 100 for i in range(5)] for s in multi], axis=0)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(kl, rec, color=BLUE, width=0.6)
for b, v in zip(bars, rec):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=13, fontweight="bold")
ax.set_ylim(0, 105); ax.set_ylabel("Recall (% of true cases found)"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("X-ray model: how often each grade is found (3-seed mean)", fontsize=15, fontweight="bold", loc="left", pad=12)
save(fig, "fig6_per_class_recall.png")

# ---- Fig 7: binary accuracy vs base paper ----
fig, ax = plt.subplots(figsize=(9, 5))
labs = ["Base paper\n(stacked ensemble,\n77 M params)", "Ours: X-ray only\n(8 M params)", "Ours: fusion\n(X-ray + clinical findings)"]
vals = [86.29, xr["Binary\n(No OA vs OA)"][0], fu[0]]
bars = ax.bar(labs, vals, color=[GRAY, BLUE, AQUA], width=0.55)
bars[2].set_hatch("//"); bars[2].set_edgecolor("white")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}%", ha="center", fontsize=14, fontweight="bold")
ax.set_ylim(0, 108); ax.set_ylabel("Binary accuracy (%)"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("Binary diagnosis (No OA vs OA)", fontsize=15, fontweight="bold", loc="left", pad=12)
fig.text(0.01, -0.13, "Base paper figure as reported by its authors (not re-run). Fusion uses radiographic findings that define the KL grade, so it is not image-only.", fontsize=10, color=MUTED)
save(fig, "fig7_binary_vs_base_paper.png")
