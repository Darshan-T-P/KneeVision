import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from kneevision.config.settings import RAW_DATA_DIR, BATCH_SIZE, MLFLOW_ENABLED, PROJECT_ROOT
from kneevision.models.image_model import load_trained_model
from kneevision.data.dataset import KneeXRayDataset
from kneevision.data.prepare import get_paths_and_labels
from kneevision.data.transforms import val_transform, tta_transforms_list
from kneevision.evaluation.report import (
    compute_metrics,
    classification_report_text,
    confusion_matrix_plot,
    write_artifacts,
)
from kneevision.training.losses import ordinal_to_class
from kneevision.utils.helpers import get_device
from kneevision.utils.logging import setup_logger
from torchvision.transforms import functional as TF

logger = setup_logger("evaluate")

if MLFLOW_ENABLED:
    from kneevision.utils.tracking import MLflowTracker
    tracker = MLflowTracker()


def apply_tta(img_tensor: torch.Tensor, tta_tfm) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    pil = TF.to_pil_image((img_tensor.cpu() * std + mean).clamp(0, 1))
    return tta_tfm(pil)


@torch.inference_mode()
def ensemble_predict(models: list[torch.nn.Module], loader: DataLoader,
                     device: torch.device, use_tta: bool = True):
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in tqdm(loader, desc="Ensemble"):
        images = images.to(device)
        probs = torch.zeros(len(images), 5, device=device)

        for model in models:
            model.eval()

            if use_tta:
                tta_probs = []
                for tta_tfm in tta_transforms_list:
                    tta_images = torch.stack([apply_tta(im, tta_tfm) for im in images]).to(device)
                    logits = model(tta_images)
                    if getattr(model, "ordinal", False):
                        class_preds = ordinal_to_class(logits)
                        p = torch.zeros(len(images), 5, device=device)
                        for c in range(5):
                            p[:, c] = (class_preds == c).float()
                    else:
                        p = F.softmax(logits, dim=1)
                    tta_probs.append(p)
                probs += torch.stack(tta_probs).mean(dim=0)
            else:
                logits = model(images)
                if getattr(model, "ordinal", False):
                    class_preds = ordinal_to_class(logits)
                    for c in range(5):
                        probs[:, c] = (class_preds == c).float()
                else:
                    probs += F.softmax(logits, dim=1)

        probs /= len(models)
        preds = probs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().numpy())
    return all_preds, all_labels, np.array(all_probs)


def main():
    device = get_device()
    logger.info("Device: %s", device)

    val_paths, val_labels = get_paths_and_labels(RAW_DATA_DIR / "test")
    val_ds = KneeXRayDataset(val_paths, val_labels, val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    logger.info("Test samples: %d", len(val_ds))

    if MLFLOW_ENABLED:
        tracker.start_run(run_name=f"evaluate_{time.strftime('%Y%m%d_%H%M%S')}", tags={"phase": "evaluation"})
        tracker.log_params({"test_samples": len(val_ds), "batch_size": BATCH_SIZE, "tta": True})

    model_names = sys.argv[1:] or ["densenet121", "vit_b_16"]
    models = []
    for name in model_names:
        candidates = [Path(f"models/best_{name}.pt"), Path(f"models/best_{name}_ordinal.pt")]
        loaded = False
        for path in candidates:
            if not path.exists():
                continue
            try:
                model = load_trained_model(path, device)
            except RuntimeError as exc:
                logger.warning("Skip %s: incompatible checkpoint (%s)", path.name, str(exc).splitlines()[0])
                continue
            n_params = sum(p.numel() for p in model.parameters())
            logger.info("Loaded %s (%s, %s params)", path.name, "ordinal" if getattr(model, "ordinal", False) else "standard", f"{n_params:,}")
            models.append(model)
            loaded = True
            break
        if not loaded:
            logger.warning("Skip %s: no loadable checkpoint", name)

    if not models:
        logger.error("No trained models found. Train first.")
        return

    preds, labels, probs = ensemble_predict(models, val_loader, device, use_tta=True)

    logger.info("=" * 60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 60)
    report = classification_report_text(labels, preds)
    logger.info("\n%s", report)

    metrics = compute_metrics(labels, preds, probs)
    logger.info("Cohen Kappa: %.4f | Accuracy: %.4f", metrics["kappa"], metrics["accuracy"])

    cm_fig = confusion_matrix_plot(labels, preds)
    cm_norm = cm_fig.axes[0].images[0].get_array()
    logger.info("\nConfusion Matrix (normalized):\n%s", np.round(cm_norm, 3))

    if MLFLOW_ENABLED:
        tracker.log_metrics(metrics)
        tracker.set_tag("num_models", len(models))
        tracker.set_tag("model_names", ",".join(m.model_name for m in models))
        tracker.set_tag("split", "test")
        tracker.set_tag("test_samples", str(len(labels)))

        artifact_dir = PROJECT_ROOT / "reports" / "evaluate"
        write_artifacts(labels, preds, probs, out_dir=artifact_dir)
        if artifact_dir.exists():
            tracker.log_artifacts(str(artifact_dir))

        if len(models) == 1:
            tracker.register_model(
                models[0],
                model_name=f"kneevision_{models[0].model_name}",
                alias="champion",
                input_example=torch.randn(1, 3, 224, 224, device=device),
            )
        tracker.end_run()


if __name__ == "__main__":
    main()