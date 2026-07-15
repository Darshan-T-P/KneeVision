import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kneevision.config.settings import RAW_DATA_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from kneevision.models.image_model import KneeXRayClassifier
from kneevision.data.dataset import KneeXRayDataset
from kneevision.data.transforms import train_transform, val_transform
from kneevision.training.trainer import train_epoch, validate
from kneevision.utils.helpers import set_seed, get_device


def get_paths_and_labels(split: str) -> tuple[list[Path], list[int]]:
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


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    train_paths, train_labels = get_paths_and_labels("train")
    val_paths, val_labels = get_paths_and_labels("val")

    print(f"Train: {len(train_paths)} images")
    print(f"Val:   {len(val_paths)} images")

    train_ds = KneeXRayDataset(train_paths, train_labels, train_transform)
    val_ds = KneeXRayDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = KneeXRayClassifier("densenet121", 5).to(device)

    class_counts = torch.tensor([2286, 1046, 1516, 757, 173], dtype=torch.float)
    class_weights = (1.0 / class_counts) * class_counts.sum() / 5
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  -> Saved best model (acc={val_acc:.4f})")


if __name__ == "__main__":
    main()
