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
    EARLY_STOP_PATIENCE, CHECKPOINT_INTERVAL, MLFLOW_ENABLED, MODELS_DIR,
)
from kneevision.models.image_model import KneeXRayClassifier, AVAILABLE_MODELS
from kneevision.data.dataset import KneeXRayDataset, MixUpDataset, make_weighted_sampler
from kneevision.data.transforms import train_transform, minority_transform, val_transform
from kneevision.training.trainer import validate, EMA, EarlyStopping, save_checkpoint, load_checkpoint
from kneevision.training.losses import FocalLoss, OrdinalLoss
from kneevision.utils.helpers import set_seed, get_device
from kneevision.utils.logging import setup_logger

logger = setup_logger("compare_models")

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


def run(model_name: str, ordinal: bool = False, resume: bool = False, batch_size: int = BATCH_SIZE):
    mode = "Ordinal" if ordinal else "Standard"
    run_name = f"{model_name}_{mode.lower()}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger.info("=" * 60)
    logger.info("Training: %s  |  %s  |  Batch: %d", model_name, mode, batch_size)
    logger.info("=" * 60)

    if MLFLOW_ENABLED:
        tracker.start_run(
            run_name=run_name,
            tags={"model": model_name, "mode": mode, "project": "kneevision"},
        )
        tracker.log_params({
            "model_name": model_name,
            "ordinal": ordinal,
            "batch_size": batch_size,
        })
        tracker.log_params_from_settings()

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
    logger.info("Parameters: %s", f"{params:,}")

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

    save_dir = MODELS_DIR
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
        logger.info("Resumed from epoch %d (best kappa=%.4f)", ep, bk)
        if MLFLOW_ENABLED:
            tracker.set_tag("resumed_from_epoch", ep)
            tracker.set_tag("resumed_best_kappa", bk)

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

        logger.info(
            "Epoch %2d | %ds | LR %.2e | Train Loss: %.4f | Val Loss: %.4f | Kappa: %.4f | EMA Kappa: %.4f",
            epoch, epoch_time, lr, train_loss, val_loss, val_kappa, ema_kappa,
        )

        if MLFLOW_ENABLED:
            tracker.log_epoch_metrics(epoch, train_loss, val_loss, val_kappa, ema_kappa, lr)

        best = max(val_kappa, ema_kappa)
        if best > best_kappa:
            best_kappa = best
            ckpt = ema.state_dict() if ema_kappa >= val_kappa else model.state_dict()
            torch.save(ckpt, best_path)
            logger.info("  -> Saved best (kappa=%.4f)", best)
            if MLFLOW_ENABLED:
                tracker.set_tag("best_kappa", best)
                tracker.set_tag("best_epoch", epoch)
                tracker.log_artifact(str(best_path))

        if epoch % CHECKPOINT_INTERVAL == 0 or early_stop.step(best):
            save_checkpoint(ckpt_path, model, optimizer, scheduler, ema,
                           early_stop, epoch, best_kappa, {}, model_name, ordinal)

        if early_stop.stopped:
            logger.info("Early stopping (no improvement for %d epochs)", EARLY_STOP_PATIENCE)
            break

    total_time = time.time() - start
    logger.info(
        "%s | Params: %s | Best Kappa: %.4f | Time: %ds",
        model_name, f"{params:,}", best_kappa, total_time,
    )

    if MLFLOW_ENABLED:
        tracker.log_metrics({"best_kappa": best_kappa, "total_time_sec": total_time, "total_params": params})
        tracker.end_run()

    return best_kappa, params, total_time


if __name__ == "__main__":
    set_seed(42)
    results = []
    for name in ["densenet121", "efficientnet-b4"]:
        kappa, params, t = run(name)
        results.append((name, params, kappa, t))

    logger.info("=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    for name, params, kappa, t in results:
        logger.info("%-20s | Params: %7s | Best Kappa: %.4f | Time: %ds", name, f"{params:,}", kappa, t)