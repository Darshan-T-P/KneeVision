import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from kneevision.config.settings import (
    RAW_DATA_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    WEIGHT_DECAY, MAX_GRAD_NORM, LABEL_SMOOTHING, MIXUP_ALPHA, SAMPLER_POWER,
    EARLY_STOP_PATIENCE, CHECKPOINT_INTERVAL,
)
from kneevision.models.image_model import KneeXRayClassifier, AVAILABLE_MODELS
from kneevision.data.dataset import KneeXRayDataset, MixUpDataset, make_weighted_sampler
from kneevision.data.transforms import train_transform, minority_transform, val_transform
from kneevision.training.trainer import validate, EMA, EarlyStopping, save_checkpoint, load_checkpoint
from kneevision.training.losses import FocalLoss, OrdinalLoss
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


def run(model_name: str, ordinal: bool = False, resume: bool = False, batch_size: int = BATCH_SIZE):
    mode = "Ordinal" if ordinal else "Standard"
    print(f"\n{'='*50}")
    print(f"Training: {model_name}  |  {mode}  |  Batch: {batch_size}")
    print(f"{'='*50}")

    device = get_device()
    IS_CUDA = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if IS_CUDA else None
    train_paths, train_labels = get_paths_and_labels("train")
    val_paths, val_labels = get_paths_and_labels("val")

    minority_set = {3, 4}

    train_ds = KneeXRayDataset(
        train_paths, train_labels,
        transform=train_transform,
        minority_transform=minority_transform,
        minority_labels=minority_set,
    )

    if MIXUP_ALPHA > 0:
        train_ds = MixUpDataset(train_ds, alpha=MIXUP_ALPHA, num_classes=5)

    sampler = make_weighted_sampler(train_labels, power=SAMPLER_POWER)
    val_ds = KneeXRayDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = KneeXRayClassifier(model_name, 5, ordinal=ordinal).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    if ordinal:
        criterion = OrdinalLoss(num_classes=5)
    else:
        class_counts = torch.tensor([2286, 1046, 1516, 757, 173], dtype=torch.float)
        class_weights = (1.0 / class_counts) * class_counts.sum() / 5
        criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0, label_smoothing=LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / (NUM_EPOCHS - warmup_epochs) * 3.14159)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ema = EMA(model, decay=0.995)

    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    suffix = "_ordinal" if ordinal else ""
    ckpt_path = save_dir / f"checkpoint_{model_name}{suffix}.pt"
    best_path = save_dir / f"best_{model_name}{suffix}.pt"

    early_stop = EarlyStopping(patience=EARLY_STOP_PATIENCE)
    start_epoch = 1
    best_kappa = -1.0

    if resume and ckpt_path.exists():
        ep, bk, hist = load_checkpoint(ckpt_path, model, optimizer, scheduler, ema, early_stop)
        start_epoch = ep + 1
        best_kappa = bk
        model = model.to(device)
        print(f"Resumed from epoch {ep} (best kappa={best_kappa:.4f})")

    start = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(images), labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(images), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
            if ema is not None:
                ema.update(model)
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        val_loss, val_kappa = validate(model, val_loader, criterion, device, use_kappa=True)
        _, ema_kappa = validate(ema.model, val_loader, criterion, device, use_kappa=True)
        scheduler.step()
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:2d} | {epoch_time:.0f}s | LR {lr:.2e} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Kappa: {val_kappa:.4f} | EMA Kappa: {ema_kappa:.4f}")

        best = max(val_kappa, ema_kappa)
        if best > best_kappa:
            best_kappa = best
            ckpt = ema.state_dict() if ema_kappa >= val_kappa else model.state_dict()
            torch.save(ckpt, best_path)
            print(f"  -> Saved best (kappa={best:.4f})")

        if epoch % CHECKPOINT_INTERVAL == 0 or early_stop.step(best):
            save_checkpoint(ckpt_path, model, optimizer, scheduler, ema,
                           early_stop, epoch, best_kappa, {}, model_name, ordinal)

        if early_stop.stopped:
            print(f"  Early stopping (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    total_time = time.time() - start
    print(f"\n{model_name} | Params: {params:,} | Best Kappa: {best_kappa:.4f} | Time: {total_time:.0f}s")
    return best_kappa, params, total_time


if __name__ == "__main__":
    set_seed(42)
    run("densenet121")
