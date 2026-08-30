"""Evaluate dedicated binary (No-OA vs OA) checkpoints saved by compare_models.py --binary.

Loads best_{name}_binary.pt (2-class heads), ensembles them with TTA,
tunes the decision threshold on the validation split, then reports test metrics.

Usage:
    uv run python scripts/evaluate_binary.py [MODEL ...] [--no-tta]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kneevision.config.settings import RAW_DATA_DIR, BATCH_SIZE, MLFLOW_ENABLED, PROJECT_ROOT
from kneevision.models.image_model import load_trained_model
from kneevision.data.dataset import KneeXRayDataset
from kneevision.data.prepare import get_paths_and_labels
from kneevision.data.transforms import build_val_transform, build_tta_transforms
from kneevision.evaluation.report import classification_report_text, write_artifacts
from kneevision.utils.helpers import get_device
from kneevision.utils.logging import setup_logger
from evaluate import apply_tta

logger = setup_logger("evaluate_binary")


@torch.inference_mode()
def predict_probs(models: list[torch.nn.Module], loader: DataLoader, device: torch.device,
                  tta_list: list, use_tta: bool = True) -> np.ndarray:
    all_probs = []
    for images, _ in loader:
        images = images.to(device)
        probs = torch.zeros(len(images), 2, device=device)
        for model in models:
            model.eval()
            if use_tta:
                tta_probs = []
                for tta_tfm in tta_list:
                    tta_images = torch.stack([apply_tta(im, tta_tfm) for im in images]).to(device)
                    tta_probs.append(F.softmax(model(tta_images), dim=1))
                probs += torch.stack(tta_probs).mean(dim=0)
            else:
                probs += F.softmax(model(images), dim=1)
        probs /= len(models)
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


def load_binary_models(model_names, device):
    models = []
    image_size = None
    for name in model_names:
        path = Path(f"models/best_{name}_binary.pt")
        if not path.exists():
            logger.warning("Skip %s: %s not found (train with compare_models.py --binary)", name, path.name)
            continue
        try:
            model = load_trained_model(path, device, num_classes=2)
        except RuntimeError as exc:
            logger.warning("Skip %s: incompatible checkpoint (%s)", path.name, str(exc).splitlines()[0])
            continue
        models.append(model)
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            size = int(json.loads(meta_path.read_text()).get("image_size") or 0)
            if size and image_size and size != image_size:
                raise SystemExit(f"Mixed resolutions in ensemble ({size} vs {image_size}); evaluate separately.")
            image_size = image_size or size or None
        logger.info("Loaded %s", path.name)
    return models, image_size


def main():
    parser = argparse.ArgumentParser(description="Evaluate dedicated binary OA-detection models")
    parser.add_argument("models", nargs="+", default=["densenet121"], help="checkpoint basenames")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    device = get_device()
    logger.info("Device: %s", device)

    test_paths, test_labels_5c = get_paths_and_labels(RAW_DATA_DIR / "test")
    val_paths, val_labels_5c = get_paths_and_labels(RAW_DATA_DIR / "val")

    models, image_size = load_binary_models(args.models, device)
    if not models:
        logger.error("No binary checkpoints found. Train first: compare_models.py --models <name> --binary")
        return
    if image_size:
        logger.info("Checkpoint resolution: %dx%d", image_size, image_size)
    val_tf = build_val_transform(image_size) if image_size else build_val_transform()
    tta_list = build_tta_transforms(image_size) if image_size else build_tta_transforms()

    test_ds = KneeXRayDataset(test_paths, test_labels_5c, val_tf)
    val_ds = KneeXRayDataset(val_paths, val_labels_5c, val_tf)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    logger.info("Val: %d | Test: %d samples", len(val_ds), len(test_ds))

    y_val = np.array([0 if grade <= 1 else 1 for grade in val_labels_5c])
    y_test = np.array([0 if grade <= 1 else 1 for grade in test_labels_5c])

    val_probs = predict_probs(models, val_loader, device, tta_list=tta_list, use_tta=not args.no_tta)
    test_probs = predict_probs(models, test_loader, device, tta_list=tta_list, use_tta=not args.no_tta)

    thresholds = np.arange(0.05, 0.96, 0.01)
    accs = [(y_val == (val_probs[:, 1] >= t).astype(int)).mean() for t in thresholds]
    best_t = float(thresholds[int(np.argmax(accs))])
    tuned_val_acc = float(max(accs))

    default_pred = (test_probs[:, 1] >= 0.5).astype(int)
    tuned_pred = (test_probs[:, 1] >= best_t).astype(int)
    default_acc = float((y_test == default_pred).mean())
    tuned_acc = float((y_test == tuned_pred).mean())

    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    auc = float(roc_auc_score(y_test, test_probs[:, 1]))

    logger.info("=" * 60)
    logger.info("DEDICATED BINARY MODEL%s (%d model%s)", " ENSEMBLE" if len(models) > 1 else "",
                len(models), "s" if len(models) > 1 else "")
    logger.info("=" * 60)
    logger.info("Default threshold 0.50 -> test accuracy: %.4f", default_acc)
    logger.info("Tuned threshold %.2f (val acc %.4f) -> test accuracy: %.4f", best_t, tuned_val_acc, tuned_acc)
    logger.info("ROC-AUC: %.4f | Balanced accuracy: %.4f", auc, balanced_accuracy_score(y_test, tuned_pred))
    logger.info("\n%s", classification_report_text(
        y_test, tuned_pred, class_names=["No OA (KL 0-1)", "OA (KL 2-4)"]))

    out_dir = PROJECT_ROOT / "reports" / "binary_dedicated"
    write_artifacts(y_test, tuned_pred, test_probs, out_dir=out_dir,
                    class_names=["No OA (KL 0-1)", "OA (KL 2-4)"])
    logger.info("Artifacts -> %s", out_dir)

    if MLFLOW_ENABLED:
        from kneevision.utils.tracking import MLflowTracker

        tracker = MLflowTracker()
        tracker.start_run(run_name=f"binary_{time.strftime('%Y%m%d_%H%M%S')}", tags={"phase": "evaluation-binary"})
        tracker.log_params({"models": ",".join(args.models), "tta": not args.no_tta})
        tracker.log_metrics({
            "accuracy": tuned_acc,
            "accuracy_default_threshold": default_acc,
            "roc_auc": auc,
            "threshold": best_t,
        })
        tracker.set_tag("split", "test")
        tracker.set_tag("model_names", ",".join(m.model_name for m in models))
        tracker.log_artifacts(str(out_dir))
        tracker.end_run()


if __name__ == "__main__":
    main()
