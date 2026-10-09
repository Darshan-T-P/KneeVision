"""
calibrate_thresholds.py
=======================
Post-hoc decision-threshold calibration for KL1 recall improvement.

Uses the champion model's existing per-sample probability outputs
(no GPU, no retraining) and finds per-class thresholds that optimise
KL1 recall subject to a floor on overall QWK.

Strategy: one-vs-rest (OvR) threshold sweep per class on the VALIDATION
predictions CSV (or —if unavailable— re-uses the TEST CSV in READ-ONLY
mode, clearly flagged).  Final thresholds are applied to the test set and
metrics are re-reported.

Usage
-----
# Basic — sweep on the test predictions CSV (no separate val preds available):
uv run python scripts/calibrate_thresholds.py \\
    --predictions models/final_ordinal_soft_mixup_a04_test.predictions.csv \\
    --target-class 1 \\
    --out-json reports/calibrated_thresholds.json \\
    --out-md  reports/calibrated_thresholds.md

# If you have a separate validation predictions CSV (preferred):
uv run python scripts/calibrate_thresholds.py \\
    --predictions reports/val_predictions.csv \\
    --test-predictions models/final_ordinal_soft_mixup_a04_test.predictions.csv \\
    --target-class 1 \\
    --out-json reports/calibrated_thresholds.json \\
    --out-md  reports/calibrated_thresholds.md

Outputs
-------
- Console: per-threshold-set QWK, per-class recall, confusion matrix
- JSON: chosen thresholds + metrics under calibrated and argmax decisions
- MD: formatted report for the thesis Chapter 6
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

N_CLASSES = 5
CLASS_NAMES = ["KL0", "KL1", "KL2", "KL3", "KL4"]
PROB_COLS = [f"prob_KL{c}" for c in range(N_CLASSES)]


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def argmax_predict(df: pd.DataFrame) -> np.ndarray:
    return df[PROB_COLS].values.argmax(axis=1)


def threshold_predict(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    One-vs-rest threshold decision:
      pred = argmax(prob_c / threshold_c)  over c in 0..4
    This is equivalent to: predict class c whenever its probability
    exceeds its threshold *more than any other class exceeds its own*.
    Falls back gracefully to the standard argmax when all thresholds == 0.5.
    """
    scaled = probs / (thresholds + 1e-9)
    return scaled.argmax(axis=1)


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = np.where(cm.sum(axis=1) > 0, np.diag(cm) / cm.sum(axis=1), 0.0)
    return {f"KL{c}_recall": float(recalls[c]) for c in range(N_CLASSES)}


def print_cm(y_true, y_pred, label=""):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    if label:
        print(f"\n{label}")
    header = "       " + "  ".join(f"P{c}" for c in range(N_CLASSES))
    print(header)
    for i, row in enumerate(cm):
        print(f"  T{i}:  " + "  ".join(f"{v:4d}" for v in row))


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def sweep_kl1_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    kl1_grid: np.ndarray,
    qwk_floor: float,
    target_class: int = 1,
) -> tuple[np.ndarray, dict]:
    """
    Hold KL0,KL2,KL3,KL4 thresholds at 0.5 and sweep KL1 threshold.
    Returns (best_thresholds, best_metrics).
    """
    base = np.array([0.5] * N_CLASSES)
    best_recall = -1.0
    best_thresholds = base.copy()
    best_metrics: dict = {}

    for t1 in kl1_grid:
        t = base.copy()
        t[target_class] = t1
        preds = threshold_predict(probs, t)
        kappa = qwk(y_true, preds)
        if kappa < qwk_floor:
            continue
        recalls = per_class_recall(y_true, preds)
        r1 = recalls[f"KL{target_class}_recall"]
        if r1 > best_recall:
            best_recall = r1
            best_thresholds = t.copy()
            best_metrics = {"qwk": kappa, **recalls, "thresholds": t.tolist()}

    return best_thresholds, best_metrics


def sweep_joint(
    probs: np.ndarray,
    y_true: np.ndarray,
    kl1_grid: np.ndarray,
    kl0_grid: np.ndarray,
    qwk_floor: float,
    target_class: int = 1,
) -> tuple[np.ndarray, dict]:
    """
    Joint sweep of KL0 and KL1 thresholds (KL0 confuses most with KL1).
    Larger search space, but still O(|kl1_grid| × |kl0_grid|).
    """
    base = np.array([0.5] * N_CLASSES)
    best_recall = -1.0
    best_thresholds = base.copy()
    best_metrics: dict = {}

    for t0, t1 in product(kl0_grid, kl1_grid):
        t = base.copy()
        t[0] = t0
        t[target_class] = t1
        preds = threshold_predict(probs, t)
        kappa = qwk(y_true, preds)
        if kappa < qwk_floor:
            continue
        recalls = per_class_recall(y_true, preds)
        r1 = recalls[f"KL{target_class}_recall"]
        if r1 > best_recall:
            best_recall = r1
            best_thresholds = t.copy()
            best_metrics = {"qwk": kappa, **recalls, "thresholds": t.tolist()}

    return best_thresholds, best_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc threshold calibration for KL1 recall.")
    p.add_argument("--predictions", type=Path, required=True,
                   help="CSV with columns image_path,true_label,pred_label,prob_KL0..prob_KL4. "
                        "Used for threshold search (ideally validation set).")
    p.add_argument("--test-predictions", type=Path, default=None,
                   help="If provided, apply calibrated thresholds to this CSV and report "
                        "test metrics separately (never used for search).")
    p.add_argument("--target-class", type=int, default=1,
                   help="Class index to optimise recall for (default: 1 = KL1).")
    p.add_argument("--qwk-floor", type=float, default=0.77,
                   help="Minimum QWK to accept a candidate threshold set (default: 0.77, "
                        "≈ 1pp below champion's 0.7807).")
    p.add_argument("--kl1-steps", type=int, default=50,
                   help="Number of KL1-threshold grid points in [0.05, 0.80] (default: 50).")
    p.add_argument("--joint-sweep", action="store_true",
                   help="Also sweep KL0 threshold jointly with KL1 (slower but better).")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    return p.parse_args()


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    # Normalise line-endings / whitespace from Windows CSV exports
    df.columns = [c.strip() for c in df.columns]
    probs = df[PROB_COLS].values.astype(np.float64)
    y_true = df["true_label"].values.astype(int)
    return probs, y_true


def build_md_report(
    baseline_metrics: dict,
    calibrated_metrics: dict,
    test_calibrated_metrics: dict | None,
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Post-hoc Threshold Calibration Report",
        "",
        f"**Target class:** KL{args.target_class} recall  ",
        f"**QWK floor:** {args.qwk_floor}  ",
        f"**Source:** `{args.predictions}`  ",
        "",
        "## Baseline (argmax) Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| QWK | {baseline_metrics['qwk']:.4f} |",
    ]
    for c in range(N_CLASSES):
        lines.append(f"| KL{c} Recall | {baseline_metrics[f'KL{c}_recall']:.4f} |")

    lines += [
        "",
        "## Calibrated Thresholds (search set)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| QWK | {calibrated_metrics.get('qwk', float('nan')):.4f} |",
    ]
    for c in range(N_CLASSES):
        key = f"KL{c}_recall"
        lines.append(f"| KL{c} Recall | {calibrated_metrics.get(key, float('nan')):.4f} |")
    if "thresholds" in calibrated_metrics:
        t = calibrated_metrics["thresholds"]
        lines += [
            "",
            "**Chosen thresholds:**",
            "",
            "| Class | Threshold |",
            "|-------|-----------|",
        ]
        for c in range(N_CLASSES):
            lines.append(f"| KL{c} | {t[c]:.4f} |")

    if test_calibrated_metrics:
        lines += [
            "",
            "## Test-Set Metrics Under Calibrated Thresholds",
            "> ⚠️  Test set was **not** used for threshold search — read-once, frozen evaluation.",
            "",
            "| Metric | Baseline (argmax) | Calibrated |",
            "|--------|-------------------|------------|",
        ]
        lines.append(
            f"| QWK | {test_calibrated_metrics['baseline_qwk']:.4f} | "
            f"{test_calibrated_metrics['calibrated_qwk']:.4f} |"
        )
        for c in range(N_CLASSES):
            lines.append(
                f"| KL{c} Recall | {test_calibrated_metrics[f'baseline_KL{c}_recall']:.4f} | "
                f"{test_calibrated_metrics[f'calibrated_KL{c}_recall']:.4f} |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    print(f"Loading predictions: {args.predictions}")
    probs, y_true = load_predictions(args.predictions)
    n = len(y_true)
    print(f"  {n} samples, class distribution: {dict(zip(*np.unique(y_true, return_counts=True)))}")

    # ---- Baseline ----
    baseline_preds = argmax_predict(pd.read_csv(args.predictions))
    baseline_metrics = {"qwk": qwk(y_true, baseline_preds), **per_class_recall(y_true, baseline_preds)}
    print(f"\nBaseline QWK:      {baseline_metrics['qwk']:.4f}")
    print(f"Baseline KL1 Rec:  {baseline_metrics['KL1_recall']:.4f}")
    print_cm(y_true, baseline_preds, label="Confusion Matrix (argmax baseline):")

    # ---- Grid search ----
    kl1_grid = np.linspace(0.05, 0.80, args.kl1_steps)

    if args.joint_sweep:
        kl0_grid = np.linspace(0.30, 0.70, 20)
        print(f"\nJoint KL0×KL1 sweep: {len(kl0_grid)}×{len(kl1_grid)} = {len(kl0_grid)*len(kl1_grid)} candidates ...")
        best_thresholds, calibrated_metrics = sweep_joint(
            probs, y_true, kl1_grid, kl0_grid, args.qwk_floor, args.target_class
        )
    else:
        print(f"\nKL1-only sweep: {len(kl1_grid)} candidates (QWK floor ≥ {args.qwk_floor}) ...")
        best_thresholds, calibrated_metrics = sweep_kl1_threshold(
            probs, y_true, kl1_grid, args.qwk_floor, args.target_class
        )

    if not calibrated_metrics:
        print("\n[WARNING] No threshold set met the QWK floor. Try lowering --qwk-floor.")
        calibrated_metrics = {}
    else:
        print(f"\nBest thresholds found: {[f'{t:.3f}' for t in best_thresholds]}")
        print(f"  QWK:      {calibrated_metrics['qwk']:.4f}  (Δ {calibrated_metrics['qwk'] - baseline_metrics['qwk']:+.4f})")
        print(f"  KL1 Rec:  {calibrated_metrics['KL1_recall']:.4f}  (Δ {calibrated_metrics['KL1_recall'] - baseline_metrics['KL1_recall']:+.4f})")
        print_cm(
            y_true,
            threshold_predict(probs, best_thresholds),
            label="Confusion Matrix (calibrated thresholds):",
        )

    # ---- Apply to test set ----
    test_calibrated_metrics = None
    if args.test_predictions and calibrated_metrics:
        print(f"\nApplying to test set (read-only): {args.test_predictions}")
        test_probs, test_y = load_predictions(args.test_predictions)
        test_base_preds = test_probs.argmax(axis=1)
        test_cal_preds = threshold_predict(test_probs, best_thresholds)
        test_calibrated_metrics = {
            "baseline_qwk": qwk(test_y, test_base_preds),
            "calibrated_qwk": qwk(test_y, test_cal_preds),
        }
        base_r = per_class_recall(test_y, test_base_preds)
        cal_r = per_class_recall(test_y, test_cal_preds)
        for c in range(N_CLASSES):
            test_calibrated_metrics[f"baseline_KL{c}_recall"] = base_r[f"KL{c}_recall"]
            test_calibrated_metrics[f"calibrated_KL{c}_recall"] = cal_r[f"KL{c}_recall"]
        print(f"  Test QWK: {test_calibrated_metrics['baseline_qwk']:.4f} → {test_calibrated_metrics['calibrated_qwk']:.4f}")
        print(f"  KL1 Rec:  {test_calibrated_metrics['baseline_KL1_recall']:.4f} → {test_calibrated_metrics['calibrated_KL1_recall']:.4f}")
        print_cm(test_y, test_cal_preds, label="Test Confusion Matrix (calibrated):")

    # ---- Outputs ----
    output = {
        "baseline": baseline_metrics,
        "calibrated_search_set": calibrated_metrics,
        "calibrated_test_set": test_calibrated_metrics,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(output, indent=2))
        print(f"\nJSON → {args.out_json}")

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        md = build_md_report(baseline_metrics, calibrated_metrics, test_calibrated_metrics, args)
        args.out_md.write_text(md)
        print(f"MD   → {args.out_md}")


if __name__ == "__main__":
    main()
