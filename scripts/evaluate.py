import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

from kneevision.config.settings import RAW_DATA_DIR, BATCH_SIZE, MLFLOW_ENABLED
from kneevision.models.image_model import KneeXRayClassifier
from kneevision.data.dataset import KneeXRayDataset
from kneevision.data.transforms import val_transform, tta_transforms_list
from kneevision.training.losses import ordinal_to_class
from kneevision.utils.helpers import get_device
from kneevision.utils.logging import setup_logger
from torchvision.transforms import functional as TF

logger = setup_logger("evaluate")

if MLFLOW_ENABLED:
    from kneevision.utils.tracking import MLflowTracker
    tracker = MLflowTracker()


def get_paths_and_labels(split: str):
    paths, labels = [], []
    split_dir = RAW_DATA_DIR / split
    for grade_dir in sorted(split_dir.iterdir()):
        if not grade_dir.is_dir():
            continue
        label = int(grade_dir.name)
        for img_path in sorted(grade_dir.glob("*.png")):
            paths.append(img_path)
            labels.append(label)
    return paths, labels


def apply_tta(img_tensor: torch.Tensor, tta_tfm) -> torch.Tensor:
    pil = TF.to_pil_image(img_tensor.cpu())
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

    val_paths, val_labels = get_paths_and_labels("test")
    val_ds = KneeXRayDataset(val_paths, val_labels, val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    logger.info("Test samples: %d", len(val_ds))

    if MLFLOW_ENABLED:
        tracker.start_run(run_name=f"evaluate_{time.strftime('%Y%m%d_%H%M%S')}", tags={"phase": "evaluation"})
        tracker.log_params({"test_samples": len(val_ds), "batch_size": BATCH_SIZE, "tta": True})

    model_names = ["densenet121", "efficientnet-b4"]
    models = []
    for name in model_names:
        path = Path(f"models/best_{name}.pt")
        if not path.exists():
            path2 = Path(f"models/best_{name}_ordinal.pt")
            if path2.exists():
                path = path2
            else:
                logger.warning("Skip %s: checkpoint not found", name)
                continue
        sd = torch.load(path, map_location=device, weights_only=True)
        last_weight = [v for k, v in sd.items() if "classifier" in k and "weight" in k][-1]
        is_ordinal = last_weight.shape[0] == 4
        model = KneeXRayClassifier(name, 5, ordinal=is_ordinal).to(device)
        model.load_state_dict(sd)
        models.append(model)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded %s (%s, %s params)", name, "ordinal" if is_ordinal else "standard", f"{n_params:,}")

    if not models:
        logger.error("No trained models found. Train first.")
        return

    preds, labels, probs = ensemble_predict(models, val_loader, device, use_tta=True)

    logger.info("=" * 60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 60)
    report = classification_report(labels, preds, digits=4)
    logger.info("\n%s", report)

    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    logger.info("Cohen Kappa: %.4f", kappa)

    cm = confusion_matrix(labels, preds)
    logger.info("\nConfusion Matrix:\n%s", cm)

    if MLFLOW_ENABLED:
        tracker.log_metrics({"test_kappa": kappa, "test_accuracy": (np.array(preds) == np.array(labels)).mean()})
        tracker.set_tag("num_models", len(models))
        tracker.end_run()


if __name__ == "__main__":
    import time
    main()