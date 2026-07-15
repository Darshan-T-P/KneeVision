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
from kneevision.data.transforms import val_transform
from kneevision.utils.helpers import get_device


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


@torch.inference_mode()
def ensemble_predict(models: list[torch.nn.Module], loader: DataLoader, device: torch.device):
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in tqdm(loader, desc="Ensemble"):
        images = images.to(device)
        probs = torch.zeros(len(images), 5, device=device)
        for model in models:
            model.eval()
            probs += F.softmax(model(images), dim=1)
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
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Test samples: {len(val_ds)}")

    model_names = ["densenet121", "efficientnet-b4"]
    models = []
    for name in model_names:
        path = Path(f"models/best_{name}.pt")
        if not path.exists():
            print(f"Skip {name}: {path} not found")
            continue
        model = KneeXRayClassifier(name, 5).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        models.append(model)
        print(f"Loaded {name} ({sum(p.numel() for p in model.parameters()):,} params)")

    if not models:
        print("No trained models found. Train first with compare_models.py")
        return

    preds, labels, probs = ensemble_predict(models, val_loader, device)

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(labels, preds, digits=4))

    print(f"Cohen Kappa: {cohen_kappa_score(labels, preds, weights='quadratic'):.4f}")

    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
