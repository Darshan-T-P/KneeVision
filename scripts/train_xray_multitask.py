"""Train the X-ray KL-grading model with auxiliary multi-task supervision.

In addition to the main 5-class KL grade, the model also predicts real
per-compartment OARSI radiographic grades (joint space narrowing, osteophytes,
subchondral sclerosis, attrition — medial/lateral) sourced from the OAI
clinical CSV (see scripts/download_oai.py). Every Kaggle X-ray image matches
an OAI patient/side record 1:1, so this uses the *same* 8,260 training images
with richer supervision, not new images. The auxiliary heads regularize the
shared backbone; only the main KL head is used at inference time.

Usage:
    uv run python scripts/train_xray_multitask.py --model densenet121 --epochs 40
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score

from kneevision.config.settings import (
    RAW_DATA_DIR, OAI_DATA_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, IMAGE_SIZE,
    WEIGHT_DECAY, MAX_GRAD_NORM, SAMPLER_POWER, EARLY_STOP_PATIENCE, MLFLOW_ENABLED, MODELS_DIR,
)
from kneevision.models.image_model import KneeXRayClassifier, AVAILABLE_MODELS
from kneevision.data.aux_dataset import KneeXRayAuxDataset, AUX_GRADE_FIELDS, AUX_NUM_GRADES, AUX_IGNORE_INDEX, collate_aux_batch
from kneevision.data.dataset import make_weighted_sampler
from kneevision.data.prepare import get_paths_and_labels, class_weights, minority_labels
from kneevision.data.transforms import train_transform, minority_transform, val_transform
from kneevision.training.trainer import EMA, EarlyStopping
from kneevision.training.losses import FocalLoss, OrdinalLoss, ordinal_to_class
from kneevision.utils.helpers import set_seed, get_device
from kneevision.utils.logging import setup_logger

logger = setup_logger("train_xray_multitask")


def load_oai_features_dict(csv_path: Path) -> dict:
    features_dict = {}
    if not csv_path.exists():
        logger.warning(f"OAI clinical CSV not found at {csv_path}; auxiliary labels will all be missing.")
        return features_dict
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            features_dict[(row["id"], row["side"])] = row
    return features_dict


def train_epoch(model, loader, criterion_main, aux_weight, device, max_grad_norm, ema, scaler, optimizer):
    model.train()
    total_loss = 0.0
    for images, labels, aux in tqdm(loader, desc="Training"):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        aux = {k: v.to(device, non_blocking=True) for k, v in aux.items()}

        optimizer.zero_grad()
        with torch.autocast("cuda", enabled=scaler is not None):
            main_logits, aux_logits = model.forward_with_aux(images)
            loss = criterion_main(main_logits, labels)
            for field in AUX_GRADE_FIELDS:
                loss = loss + aux_weight * nn.functional.cross_entropy(
                    aux_logits[field], aux[field], ignore_index=AUX_IGNORE_INDEX
                )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if ema is not None:
            ema.update(model)
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.inference_mode()
def validate(model, loader, criterion_main, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels, aux in tqdm(loader, desc="Validation"):
        images, labels = images.to(device), labels.to(device)
        main_logits = model(images)
        total_loss += criterion_main(main_logits, labels).item()

        predicted = ordinal_to_class(main_logits) if getattr(model, "ordinal", False) else main_logits.argmax(dim=1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return total_loss / len(loader), kappa


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="densenet121", choices=sorted(AVAILABLE_MODELS), help="backbone name")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--ordinal", action="store_true", help="use CORAL ordinal loss for the main KL head")
    parser.add_argument("--aux-weight", type=float, default=0.3, help="weight for the summed auxiliary losses")
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--oai-csv", type=Path, default=OAI_DATA_DIR / "processed" / "oai_clinical.csv")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    is_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if is_cuda else None
    logger.info("Device: %s | Model: %s | Aux weight: %.2f", device, args.model, args.aux_weight)

    train_paths, train_labels = get_paths_and_labels(RAW_DATA_DIR / "train")
    val_paths, val_labels = get_paths_and_labels(RAW_DATA_DIR / "val")
    logger.info("Train: %d images | Val: %d images", len(train_paths), len(val_paths))

    features_dict = load_oai_features_dict(args.oai_csv)
    matched = sum(1 for p in train_paths if _has_match(p, features_dict))
    logger.info("Auxiliary radiographic labels matched: %d/%d train images", matched, len(train_paths))

    minority_set = minority_labels(train_labels, num_classes=5)
    train_ds = KneeXRayAuxDataset(
        train_paths, train_labels, features_dict,
        transform=train_transform, minority_transform=minority_transform, minority_labels=minority_set,
    )
    val_ds = KneeXRayAuxDataset(val_paths, val_labels, features_dict, transform=val_transform)

    sampler = make_weighted_sampler(train_labels, power=SAMPLER_POWER)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2, collate_fn=collate_aux_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_aux_batch)

    model = KneeXRayClassifier(args.model, num_classes=5, ordinal=args.ordinal, aux_grades=AUX_NUM_GRADES).to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %s", f"{params:,}")

    if args.ordinal:
        criterion_main = OrdinalLoss(num_classes=5)
    else:
        weights = torch.tensor(class_weights(train_labels), dtype=torch.float)
        criterion_main = FocalLoss(alpha=weights.to(device), gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ema = EMA(model, decay=0.995)
    early_stop = EarlyStopping(patience=args.patience)

    if MLFLOW_ENABLED:
        from kneevision.utils.tracking import MLflowTracker
        tracker = MLflowTracker()
        tracker.start_run(
            run_name=f"multitask_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}",
            tags={"model": args.model, "mode": "multitask", "project": "kneevision"},
        )
        tracker.log_params({
            "model_name": args.model, "ordinal": args.ordinal, "aux_weight": args.aux_weight,
            "batch_size": args.batch_size, "aux_fields": ",".join(AUX_GRADE_FIELDS),
        })

    best_path = MODELS_DIR / f"best_{args.model}_multitask.pt"
    best_kappa = -1.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, criterion_main, args.aux_weight, device, MAX_GRAD_NORM, ema, scaler, optimizer)
        val_loss, val_kappa = validate(model, val_loader, criterion_main, device)
        _, ema_kappa = validate(ema.model, val_loader, criterion_main, device)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        logger.info(
            "Epoch %2d | %ds | Train Loss: %.4f | Val Loss: %.4f | Kappa: %.4f | EMA Kappa: %.4f",
            epoch, epoch_time, train_loss, val_loss, val_kappa, ema_kappa,
        )
        if MLFLOW_ENABLED:
            tracker.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_kappa": val_kappa, "ema_kappa": ema_kappa}, step=epoch)

        best = max(val_kappa, ema_kappa)
        if best > best_kappa:
            best_kappa = best
            ckpt = ema.state_dict() if ema_kappa >= val_kappa else model.state_dict()
            MODELS_DIR.mkdir(exist_ok=True)
            torch.save(ckpt, best_path)
            meta = {
                "image_size": IMAGE_SIZE, "model_name": args.model, "num_classes": 5,
                "ordinal": args.ordinal, "multitask": True, "aux_fields": AUX_GRADE_FIELDS,
                "best_kappa": float(best), "epoch": epoch,
            }
            best_path.with_suffix(".json").write_text(json.dumps(meta))
            logger.info("  -> Saved best (kappa=%.4f)", best)

        if early_stop.step(best):
            logger.info("Early stopping (no improvement for %d epochs)", args.patience)
            break

    total_time = time.time() - start
    logger.info("Done | Best Kappa: %.4f | Time: %ds", best_kappa, total_time)
    if MLFLOW_ENABLED:
        tracker.log_metrics({"best_kappa": best_kappa, "total_time_sec": total_time})
        tracker.end_run()


def _has_match(path: Path, features_dict: dict) -> bool:
    stem = path.stem
    if len(stem) < 2:
        return False
    pid, side_char = stem[:-1], stem[-1].upper()
    side = "left" if side_char == "L" else "right" if side_char == "R" else ""
    return (pid, side) in features_dict


if __name__ == "__main__":
    main()
