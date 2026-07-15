import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from kneevision.config.settings import RAW_DATA_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from kneevision.models.image_model import KneeXRayClassifier
from kneevision.data.dataset import KneeXRayDataset
from kneevision.data.transforms import train_transform, val_transform
from kneevision.training.trainer import train_epoch, validate
from kneevision.training.losses import FocalLoss
from kneevision.utils.helpers import set_seed, get_device


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


def run(model_name: str):
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}")

    device = get_device()
    train_paths, train_labels = get_paths_and_labels("train")
    val_paths, val_labels = get_paths_and_labels("val")

    train_ds = KneeXRayDataset(train_paths, train_labels, train_transform)
    val_ds = KneeXRayDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = KneeXRayClassifier(model_name, 5).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    class_counts = torch.tensor([2286, 1046, 1516, 757, 173], dtype=torch.float)
    class_weights = (1.0 / class_counts) * class_counts.sum() / 5
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    best_acc = 0.0
    start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch:2d} | {epoch_time:.0f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir / f"best_{model_name}.pt")
            print(f"  -> Saved (acc={val_acc:.4f})")

    total_time = time.time() - start
    print(f"\n{model_name} | Params: {params:,} | Best Acc: {best_acc:.4f} | Time: {total_time:.0f}s")
    return best_acc, params, total_time


if __name__ == "__main__":
    set_seed(42)
    results = []
    for name in ["densenet121", "efficientnet-b4"]:
        acc, params, t = run(name)
        results.append((name, params, acc, t))

    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    for name, params, acc, t in results:
        print(f"{name:20s} | Params: {params:>7,} | Best Acc: {acc:.4f} | Time: {t:.0f}s")
