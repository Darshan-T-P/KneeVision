"""Frozen held-out TEST evaluation for the CORAL/DenseNet121 ordinal models.

Single, deterministic, non-TTA forward pass over ``data/raw/test`` using the
architecture reconstructed by ``load_trained_model`` and the authoritative
metric/CM implementations in ``kneevision.evaluation.report``.

This script performs evaluation only: no gradients, no augmentation, no
sampler, no threshold tuning, no calibration, no parameter updates, and no
feedback of test metrics into training or selection.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kneevision.config.settings import RAW_DATA_DIR, BATCH_SIZE, PROJECT_ROOT
from kneevision.models.image_model import load_trained_model
from kneevision.data.dataset import KneeXRayDataset
from kneevision.data.prepare import get_paths_and_labels
from kneevision.data.transforms import val_transform
from kneevision.evaluation.report import compute_metrics, compute_confusion_matrix, classification_report_text
from kneevision.training.losses import ordinal_to_class
from kneevision.utils.helpers import get_device

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kneevision.config.settings import RAW_DATA_DIR, BATCH_SIZE, PROJECT_ROOT, MLFLOW_ENABLED
from kneevision.models.image_model import load_trained_model
from kneevision.data.dataset import KneeXRayDataset
from kneevision.data.prepare import get_paths_and_labels
from kneevision.data.transforms import val_transform
from kneevision.evaluation.report import (
    compute_metrics,
    compute_confusion_matrix,
    classification_report_text,
    confusion_matrix_plot,
    ordinal_error_plot,
)
from kneevision.training.losses import ordinal_to_class, ordinal_to_probs
from kneevision.utils.helpers import get_device

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_densenet121_ordinal.pt"
EXPECTED_SAMPLES = 1656
CLASS_NAMES = ["KL0", "KL1", "KL2", "KL3", "KL4"]


def _patient_id(path: Path) -> str:
    m = re.match(r"^(\d+)", path.stem)
    return m.group(1) if m else path.stem


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen held-out TEST evaluation for CORAL/DenseNet121 ordinal models.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Path to model checkpoint")
    parser.add_argument("--tag", type=str, default=None, help="Identifier tag for this evaluation")
    parser.add_argument("--source-run-id", type=str, default=None, help="MLflow source training run ID to associate with")
    parser.add_argument("--out-md", type=Path, default=None, help="Path to write markdown evaluation report")
    parser.add_argument("--out-json", type=Path, default=None, help="Path to write JSON evaluation metrics")
    parser.add_argument("--save-plots", action="store_true", help="Generate and save CM and ordinal error plots")
    parser.add_argument("--plots-dir", type=Path, default=None, help="Directory to save plots")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Also write a per-sample CSV (image_path, patient_id, true_label, "
                             "pred_label, prob_KL0..prob_KL4) alongside --out-json. OFF by default "
                             "so existing/default invocations are byte-for-byte unchanged; a prior "
                             "frozen evaluation run without this flag has no per-sample predictions "
                             "on disk and this script does not retroactively regenerate them for it "
                             "(that would mean re-running inference on an already-frozen result). "
                             "Pass this flag on any NEW evaluation run so future post-hoc analyses "
                             "(e.g. per-patient or per-demographic breakdowns) don't need to re-derive "
                             "everything from the aggregate confusion matrix alone.")
    parser.add_argument("--allow-non-ordinal", action="store_true",
                        help="Allow evaluation of non-ordinal (softmax/focal) checkpoints. "
                             "Uses torch.softmax for class probabilities and argmax for predictions "
                             "instead of ordinal_to_probs / ordinal_to_class. "
                             "Results are written to the same JSON/MD format for comparability.")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    sha256_before = _sha256(checkpoint_path)

    # Load companion JSON if present
    meta_json_path = checkpoint_path.with_suffix(".json")
    meta = {}
    if meta_json_path.exists():
        try:
            meta = json.loads(meta_json_path.read_text())
        except Exception as exc:
            print(f"Warning: could not parse {meta_json_path}: {exc}")

    epoch = meta.get("epoch", 23 if checkpoint_path.name == "best_densenet121_ordinal.pt" else None)
    seed = meta.get("seed", 42)
    val_qwk_selection = meta.get("best_validation_qwk", meta.get("best_kappa"))
    tag = args.tag or meta.get("tag", checkpoint_path.stem)

    device = get_device()
    model = load_trained_model(checkpoint_path, device, num_classes=5)
    is_ordinal = getattr(model, "ordinal", False)
    if not is_ordinal and not args.allow_non_ordinal:
        raise SystemExit(f"Checkpoint {checkpoint_path} was not reconstructed as an ordinal (CORAL) model. "
                         f"Pass --allow-non-ordinal to evaluate focal/softmax checkpoints.")
    model.eval()
    print(f"Loaded {checkpoint_path} ({model.model_name}, ordinal={model.ordinal}, {sum(p.numel() for p in model.parameters()):,} params) on {device}")
    print(f"Checkpoint SHA-256 before evaluation: {sha256_before}")

    test_paths, test_labels = get_paths_and_labels(RAW_DATA_DIR / "test")
    test_ds = KneeXRayDataset(test_paths, test_labels, val_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    assert len(test_paths) == EXPECTED_SAMPLES, f"expected {EXPECTED_SAMPLES} test samples, got {len(test_paths)}"
    assert len(set(test_paths)) == len(test_paths), "duplicate test sample paths detected"
    supports = np.bincount(np.asarray(test_labels), minlength=5)
    assert len(test_labels) == len(test_paths), "path/label count mismatch"

    preds, labels, probs_list = [], [], []
    for images, batch_labels in tqdm(test_loader, desc="Test inference"):
        images = images.to(device)
        logits = model(images)
        if is_ordinal:
            preds.extend(ordinal_to_class(logits).cpu().tolist())
            probs_list.extend(ordinal_to_probs(logits).cpu().tolist())
        else:
            import torch.nn.functional as F
            softmax_probs = F.softmax(logits, dim=1)
            preds.extend(softmax_probs.argmax(dim=1).cpu().tolist())
            probs_list.extend(softmax_probs.cpu().tolist())
        labels.extend(batch_labels.tolist())

    preds = np.asarray(preds)
    labels = np.asarray(labels)
    probs = np.asarray(probs_list)

    # Integrity check on checkpoint: must not have been modified
    sha256_after = _sha256(checkpoint_path)
    assert sha256_before == sha256_after, "FATAL: Checkpoint file was modified during evaluation!"

    metrics = compute_metrics(labels, preds, probs=probs, num_classes=5)
    cm = compute_confusion_matrix(labels, preds, num_classes=5)

    assert len(preds) == EXPECTED_SAMPLES, "prediction count mismatch"
    assert cm.sum() == EXPECTED_SAMPLES, f"confusion-matrix total {cm.sum()} != {EXPECTED_SAMPLES}"
    assert cm.shape == (5, 5) and cm.round(9).min() >= 0
    row_sums = cm.sum(axis=1)
    assert np.array_equal(row_sums, supports), "CM row sums != test class counts"

    report_text = classification_report_text(labels, preds, class_names=CLASS_NAMES, num_classes=5)

    per_class = [
        {
            "class": cn,
            "precision": metrics[f"{cn}_precision"] if f"{cn}_precision" in metrics else metrics[f"KL {i}_precision"],
            "recall": metrics[f"{cn}_recall"] if f"{cn}_recall" in metrics else metrics[f"KL {i}_recall"],
            "f1": metrics[f"{cn}_f1"] if f"{cn}_f1" in metrics else metrics[f"KL {i}_f1"],
            "support": int(supports[i]),
        }
        for i, cn in enumerate(CLASS_NAMES)
    ]

    train_patients = {_patient_id(p) for p in get_paths_and_labels(RAW_DATA_DIR / "train")[0]}
    val_patients = {_patient_id(p) for p in get_paths_and_labels(RAW_DATA_DIR / "val")[0]}
    test_patients = {_patient_id(p) for p in test_paths}

    # Error analysis calculations
    errors = np.abs(labels.astype(int) - preds.astype(int))
    error_dist = {int(k): int((errors == k).sum()) for k in range(5)}
    error_dist_pct = {int(k): float((errors == k).mean() * 100) for k in range(5)}

    error_analysis = {
        "kl2_to_kl3_error": int(cm[2, 3]),
        "kl3_to_kl2_error": int(cm[3, 2]),
        "extreme_error_kl0_to_kl4": int(cm[0, 4]),
        "extreme_error_kl4_to_kl0": int(cm[4, 0]),
        "error_distribution_counts": error_dist,
        "error_distribution_percentages": error_dist_pct,
        "exact_accuracy_pct": float((errors == 0).mean() * 100),
        "within_1_pct": float((errors <= 1).mean() * 100),
        "within_2_pct": float((errors <= 2).mean() * 100),
    }

    selection_desc = (
        f"val_qwk {val_qwk_selection} (raw, epoch {epoch}); no test data used for selection"
        if val_qwk_selection is not None
        else "validation rule; no test data used for selection"
    )

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_before,
        "epoch": epoch,
        "seed": seed,
        "tag": tag,
        "is_ordinal": is_ordinal,
        "source_run_id": args.source_run_id,
        "split": "test",
        "sample_count": int(len(test_paths)),
        "class_counts": [int(s) for s in supports],
        "unique_patients": int(len(test_patients)),
        "patient_overlap": {
            "train": int(len(train_patients & test_patients)),
            "val": int(len(val_patients & test_patients)),
        },
        "qwk": metrics["qwk"],
        "mae": metrics["mae"],
        "within_1": metrics["within_1_accuracy"],
        "within_2": metrics["within_2_accuracy"],
        "accuracy": metrics["exact_accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "linear_kappa": metrics["kappa_linear"],
        "ece": metrics.get("ece"),
        "brier_macro": metrics.get("brier_macro"),
        "brier_multiclass": metrics.get("brier_multiclass"),
        "brier_per_class": {
            f"KL{c}": metrics.get(f"brier_KL{c}") for c in range(5) if f"brier_KL{c}" in metrics
        },
        "per_class": per_class,
        "confusion_matrix": [row.tolist() for row in cm],
        "class_order": CLASS_NAMES,
        "error_analysis": error_analysis,
        "selection": selection_desc,
    }

    m = metrics
    print(f"\n========== HELD-OUT TEST EVALUATION - {tag} (epoch {epoch}, seed {seed}) ==========")
    print(f"QWK: {m['qwk']:.4f}  |  MAE: {m['mae']:.4f}")
    print(f"Within-1: {m['within_1_accuracy']:.4f}  |  Within-2: {m['within_2_accuracy']:.4f}")
    print(f"Exact accuracy: {m['exact_accuracy']:.4f}  |  Linear kappa: {m['kappa_linear']:.4f}")
    print(f"Macro P/R/F1: {m['macro_precision']:.4f}/{m['macro_recall']:.4f}/{m['macro_f1']:.4f}  |  Weighted F1: {m['weighted_f1']:.4f}")
    if "ece" in m:
        print(f"ECE: {m['ece']:.4f}  |  Brier (macro): {m.get('brier_macro', 0.0):.4f}  |  Brier (multiclass): {m.get('brier_multiclass', 0.0):.4f}")
    print(f"\n{report_text}")
    hdr = " ".join(f"{c:>6}" for c in CLASS_NAMES)
    print("Confusion matrix (rows=true, cols=predicted):")
    print("        " + hdr)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]}: " + " ".join(f"{v:>6}" for v in row) + f"  |true={row.sum()}")
    print("  pred: " + " ".join(f"{cm[:, c].sum():>3}" for c in range(5)) + "  |total=" + str(int(cm.sum())))
    print("---------------------------------------------------------------------------------------")
    print(f"KL2 -> KL3 errors: {error_analysis['kl2_to_kl3_error']} | KL3 -> KL2 errors: {error_analysis['kl3_to_kl2_error']}")
    print(f"Extreme errors: KL0->KL4 = {error_analysis['extreme_error_kl0_to_kl4']}, KL4->KL0 = {error_analysis['extreme_error_kl4_to_kl0']}")
    print(f"Absolute error distribution: {error_dist_pct}")
    print("=======================================================================================")

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    out_md = args.out_md
    if out_md is None:
        if checkpoint_path.name == "best_densenet121_ordinal.pt":
            out_md = reports_dir / "CORAL_DENSENET121_TEST_EVALUATION.md"
        else:
            out_md = reports_dir / f"{tag.upper()}_EVALUATION.md"

    out_json = args.out_json
    if out_json is None:
        if checkpoint_path.name == "best_densenet121_ordinal.pt":
            out_json = PROJECT_ROOT / "models" / "best_densenet121_ordinal.test.json"
        else:
            out_json = PROJECT_ROOT / "models" / f"{tag}.json"

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text(_markdown(report, metrics, cm, report_text, error_analysis))
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_md} and {out_json}")

    if args.save_predictions:
        import csv
        pred_csv = out_json.with_suffix("").with_suffix(".predictions.csv")
        with open(pred_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "patient_id", "true_label", "pred_label"]
                             + [f"prob_{cn}" for cn in CLASS_NAMES])
            for path_i, label_i, pred_i, probs_i in zip(test_paths, labels.tolist(), preds.tolist(), probs.tolist()):
                writer.writerow([str(path_i), _patient_id(path_i), label_i, pred_i] + probs_i)
        print(f"Wrote {pred_csv} (per-sample predictions for future post-hoc analysis)")

    plot_paths = []
    if args.save_plots:
        plots_dir = args.plots_dir or (reports_dir / tag)
        plots_dir.mkdir(parents=True, exist_ok=True)
        cm_png = plots_dir / "confusion_matrix.png"
        err_png = plots_dir / "ordinal_error_distribution.png"
        confusion_matrix_plot(labels, preds, class_names=CLASS_NAMES, out_path=cm_png)
        ordinal_error_plot(labels, preds, out_path=err_png)
        plot_paths.extend([cm_png, err_png])
        print(f"Saved plots to {plots_dir}")

    # MLflow association
    if args.source_run_id and MLFLOW_ENABLED:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        try:
            # Set evaluation tags
            client.set_tag(args.source_run_id, "test_evaluation_tag", tag)
            client.set_tag(args.source_run_id, "test_evaluation_status", "COMPLETED")
            client.set_tag(args.source_run_id, "test_split", "test")
            client.set_tag(args.source_run_id, "test_samples", str(len(test_paths)))
            client.set_tag(args.source_run_id, "test_checkpoint_sha256", sha256_before)
            client.set_tag(args.source_run_id, "test_qwk", str(round(metrics["qwk"], 6)))

            # Log test metrics with test_ prefix
            test_metrics_to_log = {
                "test_qwk": metrics["qwk"],
                "test_mae": metrics["mae"],
                "test_within_1_accuracy": metrics["within_1_accuracy"],
                "test_within_2_accuracy": metrics["within_2_accuracy"],
                "test_exact_accuracy": metrics["exact_accuracy"],
                "test_macro_precision": metrics["macro_precision"],
                "test_macro_recall": metrics["macro_recall"],
                "test_macro_f1": metrics["macro_f1"],
                "test_weighted_f1": metrics["weighted_f1"],
                "test_linear_kappa": metrics["kappa_linear"],
            }
            if "ece" in metrics and metrics["ece"] is not None:
                test_metrics_to_log["test_ece"] = float(metrics["ece"])
            if "brier_macro" in metrics and metrics["brier_macro"] is not None:
                test_metrics_to_log["test_brier_macro"] = float(metrics["brier_macro"])
            if "brier_multiclass" in metrics and metrics["brier_multiclass"] is not None:
                test_metrics_to_log["test_brier_multiclass"] = float(metrics["brier_multiclass"])

            for k, v in test_metrics_to_log.items():
                client.log_metric(args.source_run_id, k, v)

            # Log artifacts to MLflow run
            client.log_artifact(args.source_run_id, str(out_json))
            client.log_artifact(args.source_run_id, str(out_md))
            for p in plot_paths:
                client.log_artifact(args.source_run_id, str(p))
            print(f"Successfully associated test metrics and artifacts with MLflow run {args.source_run_id}")
        except Exception as exc:
            print(f"Warning: MLflow logging failed: {exc}")


def _markdown(report: dict, m: dict, cm: np.ndarray, report_text: str, error_analysis: dict) -> str:
    lines = [
        f"# {report.get('tag', 'CORAL/DenseNet121')} — Frozen Held-Out TEST Evaluation",
        "",
        "## Experiment identity",
        f"- Model: DenseNet121 + {'CORAL ordinal head' if report.get('is_ordinal') else 'focal loss / softmax head'}",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Checkpoint SHA-256: `{report.get('checkpoint_sha256', 'N/A')}`",
        f"- Training epoch: {report['epoch']}",
        f"- Seed: {report['seed']}",
        f"- Selection criteria: {report['selection']}",
        "- Selected weights: raw",
        f"- Test samples: {report['sample_count']}",
        "",
        "## Test results",
        "| Metric | Value |",
        "|---|---|",
        f"| QWK | {m['qwk']:.6f} |",
        f"| MAE | {m['mae']:.6f} |",
        f"| Within-1 accuracy | {m['within_1_accuracy']:.6f} |",
        f"| Within-2 accuracy | {m['within_2_accuracy']:.6f} |",
        f"| Exact accuracy | {m['exact_accuracy']:.6f} |",
        f"| Macro precision | {m['macro_precision']:.6f} |",
        f"| Macro recall | {m['macro_recall']:.6f} |",
        f"| Macro F1 | {m['macro_f1']:.6f} |",
        f"| Weighted F1 | {m['weighted_f1']:.6f} |",
        f"| Linear weighted kappa | {m['kappa_linear']:.6f} |",
    ]
    if "ece" in m and m["ece"] is not None:
        lines.append(f"| Expected Calibration Error (ECE) | {m['ece']:.6f} |")
    if "brier_macro" in m and m["brier_macro"] is not None:
        lines.append(f"| Brier score (macro, mean OvR) | {m['brier_macro']:.6f} |")
    if "brier_multiclass" in m and m["brier_multiclass"] is not None:
        lines.append(f"| Brier score (multiclass, sum OvR) | {m['brier_multiclass']:.6f} |")
    lines += [
        "",
        "## Per-class results (KL0–KL4)",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for row in report["per_class"]:
        lines.append(f"| {row['class']} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['support']} |")
    lines += [
        "",
        "## Confusion matrix (rows = true KL, columns = predicted KL)",
    ]
    hdr = "| | " + " | ".join(report["class_order"]) + " | true |"
    lines.append(hdr)
    lines.append("|" + "---|" * 6)
    for i, row in enumerate(cm):
        lines.append(
            f"| {report['class_order'][i]} | " + " | ".join(str(v) for v in row) + f" | {int(row.sum())} |"
        )
    lines.append(
        "| predicted | " + " | ".join(str(int(cm[:, c].sum())) for c in range(5)) + f" | {int(cm.sum())} |"
    )
    lines += [
        "",
        "## Detailed error analysis",
        f"- **KL2 → KL3 errors**: {error_analysis['kl2_to_kl3_error']} cases",
        f"- **KL3 → KL2 errors**: {error_analysis['kl3_to_kl2_error']} cases",
        f"- **Extreme error KL0 → KL4**: {error_analysis['extreme_error_kl0_to_kl4']} cases",
        f"- **Extreme error KL4 → KL0**: {error_analysis['extreme_error_kl4_to_kl0']} cases",
        f"- **Percentage exactly correct**: {error_analysis['exact_accuracy_pct']:.2f}%",
        f"- **Percentage within ±1 grade**: {error_analysis['within_1_pct']:.2f}%",
        f"- **Percentage within ±2 grades**: {error_analysis['within_2_pct']:.2f}%",
        "",
        "### Absolute ordinal error distribution (|True − Predicted|):",
        "| Error Distance | Count | Percentage |",
        "|---|---|---|",
    ]
    for dist, cnt in error_analysis["error_distribution_counts"].items():
        pct = error_analysis["error_distribution_percentages"][dist]
        lines.append(f"| {dist} | {cnt} | {pct:.2f}% |")

    lines += [
        "",
        "## Test-set integrity",
        f"- Test sample count: {report['sample_count']}",
        f"- Test class counts: {report['class_counts']}",
        f"- Unique patients: {report['unique_patients']}",
        f"- Patient overlap with train: {report['patient_overlap']['train']} (must be 0), with val: {report['patient_overlap']['val']} (must be 0)",
        "- No test labels were used to select the checkpoint.",
        "- No test metric was fed back into training.",
        "- No threshold or hyperparameter was changed after seeing test results.",
        "- Checkpoint verified unchanged before and after evaluation.",
        "",
        "## Integrity statement",
        f"This is a frozen held-out evaluation. The checkpoint `{report['checkpoint']}` "
        f"(epoch {report['epoch']}, seed {report['seed']}, raw weights, validation selection: {report['selection']}) "
        "was selected exclusively using validation QWK. The test split was held out during training and model selection "
        "and used only for this final, single-pass evaluation. Test results were NOT used for model "
        "selection or any form of tuning.",
        "",
        "## Classification report",
        "```",
        report_text,
        "```",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()