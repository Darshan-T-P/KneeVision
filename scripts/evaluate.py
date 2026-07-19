import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

from kneevision.config.settings import RAW_DATA_DIR, BATCH_SIZE
from kneevision.models.image_model import KneeXRayClassifier
from kneevision.data.dataset import KneeXRayDataset
from kneevision.data.transforms import val_transform, tta_transforms_list
from kneevision.training.losses import ordinal_to_class
from kneevision.utils.helpers import get_device
from torchvision.transforms import functional as TF
from PIL import Image


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
    print(f"Device: {device}")

    val_paths, val_labels = get_paths_and_labels("test")
    val_ds = KneeXRayDataset(val_paths, val_labels, val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Test samples: {len(val_ds)}")

    model_names = ["densenet121", "efficientnet-b4"]
    models = []
    for name in model_names:
        path = Path(f"models/best_{name}.pt")
        if not path.exists():
            path2 = Path(f"models/best_{name}_ordinal.pt")
            if path2.exists():
                path = path2
            else:
                print(f"Skip {name}: checkpoint not found")
                continue
        sd = torch.load(path, map_location=device, weights_only=True)
        last_weight = [v for k, v in sd.items() if "classifier" in k and "weight" in k][-1]
        is_ordinal = last_weight.shape[0] == 4
        model = KneeXRayClassifier(name, 5, ordinal=is_ordinal).to(device)
        model.load_state_dict(sd)
        models.append(model)
        print(f"Loaded {name} ({'ordinal' if is_ordinal else 'standard'}, {sum(p.numel() for p in model.parameters()):,} params)")

    if not models:
        print("No trained models found. Train first.")
        return

    preds, labels, probs = ensemble_predict(models, val_loader, device, use_tta=True)

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(labels, preds, digits=4))

    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    print(f"Cohen Kappa: {kappa:.4f}")

    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
