import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kneevision.config.settings import RAW_DATA_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from kneevision.models.image_model import KneeXRayClassifier
from kneevision.data.dataset import KneeXRayDataset
from kneevision.data.prepare import get_paths_and_labels, class_weights
from kneevision.data.transforms import train_transform, val_transform
from kneevision.training.trainer import train_epoch, validate
from kneevision.utils.helpers import set_seed, get_device
from kneevision.utils.logging import setup_logger

logger = setup_logger("train_xray")


def main():
    set_seed(42)
    device = get_device()
    logger.info("Device: %s", device)

    train_paths, train_labels = get_paths_and_labels(RAW_DATA_DIR / "train")
    val_paths, val_labels = get_paths_and_labels(RAW_DATA_DIR / "val")

    logger.info("Train: %d images", len(train_paths))
    logger.info("Val:   %d images", len(val_paths))

    train_ds = KneeXRayDataset(train_paths, train_labels, train_transform)
    val_ds = KneeXRayDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = KneeXRayClassifier("densenet121", 5).to(device)

    weights = torch.tensor(class_weights(train_labels), dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch %2d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f",
            epoch, train_loss, val_loss, val_acc,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            logger.info("  -> Saved best model (acc=%.4f)", val_acc)


if __name__ == "__main__":
    main()