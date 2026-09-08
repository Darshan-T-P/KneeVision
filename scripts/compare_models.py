import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader
import time

from kneevision.config.settings import (
    RAW_DATA_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, IMAGE_SIZE,
    WEIGHT_DECAY, MAX_GRAD_NORM, LABEL_SMOOTHING, MIXUP_ALPHA, SAMPLER_POWER,
    EARLY_STOP_PATIENCE, CHECKPOINT_INTERVAL, MLFLOW_ENABLED, MODELS_DIR,
)
from kneevision.models.image_model import KneeXRayClassifier
from kneevision.data.dataset import KneeXRayDataset, MixUpDataset, make_weighted_sampler
from kneevision.data.prepare import get_paths_and_labels, class_weights, minority_labels
from kneevision.data.transforms import train_transform, minority_transform, val_transform
from kneevision.training.trainer import train_epoch, validate, EMA, EarlyStopping, save_checkpoint, load_checkpoint
from kneevision.training.losses import FocalLoss, OrdinalLoss
from kneevision.utils.helpers import set_seed, get_device
from kneevision.utils.logging import setup_logger

logger = setup_logger("compare_models")

if MLFLOW_ENABLED:
    from kneevision.utils.tracking import MLflowTracker
    tracker = MLflowTracker()


def map_binary(labels: list[int]) -> list[int]:
    """KL 0-1 -> 0 (No OA), KL 2-4 -> 1 (radiographic OA)."""
    return [0 if grade <= 1 else 1 for grade in labels]


def run(model_name: str, ordinal: bool = False, resume: bool = False, batch_size: int = BATCH_SIZE, num_epochs: int = NUM_EPOCHS, binary: bool = False, patience: int = EARLY_STOP_PATIENCE):
    num_classes = 2 if binary else 5
    mode = "Binary" if binary else ("Ordinal" if ordinal else "Standard")
    run_name = f"{model_name}_{mode.lower()}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger.info("=" * 60)
    logger.info("Training: %s  |  %s  | %d classes | Batch: %d", model_name, mode, num_classes, batch_size)
    logger.info("=" * 60)

    if MLFLOW_ENABLED:
        tracker.start_run(
            run_name=run_name,
            tags={"model": model_name, "mode": mode, "project": "kneevision"},
        )
        tracker.log_params({
            "model_name": model_name,
            "ordinal": ordinal,
            "binary": binary,
            "batch_size": batch_size,
        })
        tracker.log_params_from_settings()

    device = get_device()
    IS_CUDA = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if IS_CUDA else None
    train_paths, train_labels = get_paths_and_labels(RAW_DATA_DIR / "train")
    val_paths, val_labels = get_paths_and_labels(RAW_DATA_DIR / "val")

    if binary:
        train_labels = map_binary(train_labels)
        val_labels = map_binary(val_labels)

    minority_set = minority_labels(train_labels, num_classes=num_classes)

    train_ds = KneeXRayDataset(
        train_paths, train_labels,
        transform=train_transform,
        minority_transform=minority_transform,
        minority_labels=minority_set,
    )

    if MIXUP_ALPHA > 0:
        train_ds = MixUpDataset(train_ds, alpha=MIXUP_ALPHA, num_classes=num_classes)

    sampler = make_weighted_sampler(train_labels, power=SAMPLER_POWER)
    val_ds = KneeXRayDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = KneeXRayClassifier(model_name, num_classes, ordinal=ordinal).to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %s", f"{params:,}")

    if ordinal:
        criterion = OrdinalLoss(num_classes=num_classes)
    else:
        class_weights_t = torch.tensor(class_weights(train_labels, num_classes=num_classes), dtype=torch.float)
        criterion = FocalLoss(alpha=class_weights_t.to(device), gamma=2.0, label_smoothing=LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    warmup_epochs = min(5, max(1, num_epochs // 6))
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        span = max(1, num_epochs - warmup_epochs)
        return 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / span * 3.14159)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ema = EMA(model, decay=0.995)

    save_dir = MODELS_DIR
    save_dir.mkdir(exist_ok=True)
    suffix = "_ordinal" if ordinal else ("_binary" if binary else "")
    ckpt_path = save_dir / f"checkpoint_{model_name}{suffix}.pt"
    best_path = save_dir / f"best_{model_name}{suffix}.pt"

    early_stop = EarlyStopping(patience=patience)
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

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            max_grad_norm=MAX_GRAD_NORM, ema=ema, scaler=scaler,
        )
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
            meta = {
                "image_size": IMAGE_SIZE,
                "model_name": model_name,
                "num_classes": num_classes,
                "binary": binary,
                "ordinal": ordinal,
                "best_kappa": float(best),
                "epoch": epoch,
            }
            best_path.with_suffix(".json").write_text(json.dumps(meta))
            logger.info("  -> Saved best (kappa=%.4f)", best)
            if MLFLOW_ENABLED:
                tracker.set_tag("best_kappa", best)
                tracker.set_tag("best_epoch", epoch)
                tracker.log_artifact(str(best_path))

        stopped_now = early_stop.step(best)
        if epoch % CHECKPOINT_INTERVAL == 0 or stopped_now:
            save_checkpoint(ckpt_path, model, optimizer, scheduler, ema,
                           early_stop, epoch, best_kappa, {}, model_name, ordinal)

        if early_stop.stopped:
            logger.info("Early stopping (no improvement for %d epochs)", patience)
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
    import argparse

    parser = argparse.ArgumentParser(description="Train and compare X-ray backbones")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="max training epochs (default: %(default)s)")
    parser.add_argument("--models", nargs="+", default=["densenet121", "efficientnet-b4"],
                        metavar="BACKBONE", help="backbones to train")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--binary", action="store_true",
                        help="train a 2-class head (KL0-1 vs KL2-4) instead of 5-class KL grading")
    parser.add_argument("--ordinal", action="store_true",
                        help="use CORAL ordinal loss/decoding instead of Focal loss (KL grades are ordered)")
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE,
                        help="early-stopping patience in epochs (default: %(default)s)")
    args = parser.parse_args()

    set_seed(42)
    logger.info("Epochs: %d | Models: %s | Binary: %s | Ordinal: %s | Patience: %d",
                args.epochs, ", ".join(args.models), args.binary, args.ordinal, args.patience)
    results = []
    for name in args.models:
        kappa, params, t = run(name, ordinal=args.ordinal, batch_size=args.batch_size, num_epochs=args.epochs, binary=args.binary, patience=args.patience)
        results.append((name, params, kappa, t))

    logger.info("=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    for name, params, kappa, t in results:
        logger.info("%-20s | Params: %7s | Best Kappa: %.4f | Time: %ds", name, f"{params:,}", kappa, t)