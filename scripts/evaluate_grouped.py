"""Evaluate trained 5-class models under clinically grouped label schemes.

Groups the 5-class softmax outputs post-hoc (no retraining):
  binary : KL0-1 -> No OA | KL2-4 -> OA   (KL>=2 = radiographic OA)
  3class : KL0-1 None/Doubtful | KL2 Mild | KL3-4 Moderate/Severe

Usage:
    uv run python scripts/evaluate_grouped.py [MODEL ...] [--groups binary 3class] [--no-tta]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from torch.utils.data import DataLoader

from kneevision.config.settings import RAW_DATA_DIR, BATCH_SIZE, MLFLOW_ENABLED, PROJECT_ROOT
from kneevision.models.image_model import load_trained_model
from kneevision.data.dataset import KneeXRayDataset
from kneevision.data.prepare import get_paths_and_labels
from kneevision.data.transforms import val_transform
from kneevision.evaluation.report import (
    classification_report_text,
    write_artifacts,
)
from kneevision.utils.helpers import get_device
from kneevision.utils.logging import setup_logger
from evaluate import ensemble_predict

logger = setup_logger("evaluate_grouped")

GROUPINGS: dict[str, dict[str, tuple[int, ...]]] = {
    "binary": {
        "names": ["No OA (KL 0-1)", "OA (KL 2-4)"],
        "members": [(0, 1), (2, 3, 4)],
    },
    "3class": {
        "names": ["None/Doubtful (KL 0-1)", "Mild (KL 2)", "Moderate/Severe (KL 3-4)"],
        "members": [(0, 1), (2,), (3, 4)],
    },
}


def group_labels(labels, members):
    lookup = {}
    for gid, grades in enumerate(members):
        for grade in grades:
            lookup[grade] = gid
    return np.array([lookup[int(label)] for label in labels])


def group_probs(probs: np.ndarray, members) -> np.ndarray:
    return np.stack([probs[:, list(grades)].sum(axis=1) for grades in members], axis=1)


def load_models(model_names, device):
    models = []
    for name in model_names:
        candidates = [Path(f"models/best_{name}.pt"), Path(f"models/best_{name}_ordinal.pt")]
        for path in candidates:
            if not path.exists():
                continue
            try:
                model = load_trained_model(path, device)
            except RuntimeError as exc:
                logger.warning("Skip %s: incompatible checkpoint (%s)", path.name, str(exc).splitlines()[0])
                continue
            models.append(model)
            logger.info("Loaded %s", path.name)
            break
        else:
            logger.warning("Skip %s: no loadable checkpoint", name)
    return models


def main():
    parser = argparse.ArgumentParser(description="Evaluate grouped-class performance")
    parser.add_argument("models", nargs="+", default=["densenet121"], help="checkpoint basenames")
    parser.add_argument("--groups", nargs="+", choices=list(GROUPINGS), default=["binary", "3class"])
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    device = get_device()
    logger.info("Device: %s", device)

    paths, labels_5c = get_paths_and_labels(RAW_DATA_DIR / "test")
    ds = KneeXRayDataset(paths, labels_5c, val_transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    logger.info("Test samples: %d", len(ds))

    val_paths, val_labels_5c = get_paths_and_labels(RAW_DATA_DIR / "val")
    val_ds = KneeXRayDataset(val_paths, val_labels_5c, val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    logger.info("Val samples (threshold tuning): %d", len(val_ds))

    models = load_models(args.models, device)
    if not models:
        logger.error("No trained models found. Train first.")
        return

    _, labels_5c_true, probs_5c = ensemble_predict(models, loader, device, use_tta=not args.no_tta)
    _, val_labels_5c_true, val_probs_5c = ensemble_predict(models, val_loader, device, use_tta=not args.no_tta)

    if MLFLOW_ENABLED:
        from kneevision.utils.tracking import MLflowTracker

        tracker = MLflowTracker()
        tracker.start_run(run_name=f"grouped_{time.strftime('%Y%m%d_%H%M%S')}", tags={"phase": "evaluation-grouped"})
        tracker.log_params({"models": ",".join(args.models), "tta": not args.no_tta})

    for group_name in args.groups:
        spec = GROUPINGS[group_name]
        names = spec["names"]
        n = len(names)

        y_true = group_labels(labels_5c_true, spec["members"])
        gprobs = group_probs(probs_5c, spec["members"])

        if n == 2:
            val_y = group_labels(val_labels_5c_true, spec["members"])
            val_p = group_probs(val_probs_5c, spec["members"])[:, 1]
            thresholds = np.arange(0.05, 0.96, 0.01)
            accs = [(val_y == (val_p >= t).astype(int)).mean() for t in thresholds]
            best_t = float(thresholds[int(np.argmax(accs))])
            default_acc = float((y_true == (gprobs[:, 1] >= 0.5).astype(int)).mean())
            y_pred = (gprobs[:, 1] >= best_t).astype(int)
            tuned_val_acc = float(max(accs))
        else:
            best_t = None
            y_pred = gprobs.argmax(axis=1)

        logger.info("=" * 60)
        logger.info("%s GROUPING (%d classes)", group_name.upper(), n)
        logger.info("=" * 60)

        if n == 2:
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(y_true, gprobs[:, 1])
            test_acc = float((y_true == y_pred).mean())
            logger.info("Default threshold 0.50 -> test accuracy: %.4f", default_acc)
            logger.info("Tuned threshold %.2f (val acc %.4f) -> test accuracy: %.4f | ROC-AUC: %.4f",
                        best_t, tuned_val_acc, test_acc, auc)
        else:
            logger.info("\n%s", classification_report_text(y_true, y_pred, class_names=names))
            from sklearn.metrics import cohen_kappa_score

            kappa_q = cohen_kappa_score(y_true, y_pred, weights="quadratic")
            acc = float((y_true == y_pred).mean())
            logger.info("Accuracy: %.4f | Quadratic Kappa: %.4f", acc, kappa_q)

        out_dir = PROJECT_ROOT / "reports" / f"grouped_{group_name}"
        artifacts = write_artifacts(y_true, y_pred, gprobs, out_dir=out_dir, class_names=names)
        if artifacts["report_html"]:
            logger.info("Artifacts -> %s", out_dir)

        if MLFLOW_ENABLED:
            final_acc = test_acc if n == 2 else acc
            metrics = {f"{group_name}_accuracy": final_acc}
            if n == 2:
                metrics[f"{group_name}_roc_auc"] = float(auc)
                metrics[f"{group_name}_threshold"] = best_t
            else:
                metrics[f"{group_name}_kappa"] = kappa_q
            tracker.log_metrics(metrics)
            tracker.log_artifacts(str(out_dir))

    if MLFLOW_ENABLED:
        tracker.set_tag("split", "test")
        tracker.set_tag("model_names", ",".join(m.model_name for m in models))
        tracker.end_run()


if __name__ == "__main__":
    main()
