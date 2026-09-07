import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, cohen_kappa_score

from kneevision.config.settings import (
    CLINICAL_DATA_DIR, CLINICAL_MAX_LENGTH, CLINICAL_BATCH_SIZE,
    MODELS_DIR, MLFLOW_ENABLED,
)
from kneevision.clinical.model import ClinicalTextModel
from kneevision.clinical.dataset import ClinicalTextDataset
from kneevision.clinical.prepare import (
    load_reports_from_folders, load_reports_csv,
    build_synthetic_splits,
)
from kneevision.data.prepare import get_splits, class_weights
from kneevision.utils.helpers import set_seed, get_device
from kneevision.utils.logging import setup_logger
from kneevision.training.losses import FocalLoss

logger = setup_logger("train_clinical")

if MLFLOW_ENABLED:
    from kneevision.utils.tracking import MLflowTracker
    tracker = MLflowTracker()


def validate(model, loader, criterion, device, is_ordinal=False) -> tuple[float, float, list, list]:
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            if is_ordinal:
                from kneevision.training.losses import ordinal_to_class
                all_preds.extend(ordinal_to_class(logits).cpu().tolist())
            else:
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return total_loss / len(loader), acc, kappa, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train clinical text model (BioClinicalBERT)")
    parser.add_argument("--data", type=Path, default=CLINICAL_DATA_DIR,
                        help="Directory with {split}/{kl_grade}/*.txt reports, or a CSV with a split column")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic reports from the image dataset (fallback until real reports exist)")
    parser.add_argument("--image-data", type=Path, default=None, help="Image raw dir used by --synthetic")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=CLINICAL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--freeze", action="store_true", help="Freeze BioClinicalBERT encoder")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--ordinal", action="store_true", help="Use ordinal regression")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (epochs without val kappa improvement)")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    logger.info("Device: %s", device)

    if args.synthetic:
        image_root = args.image_data or Path("data/raw")
        image_splits = get_splits(image_root)
        if not image_splits:
            logger.error("No image splits found at %s", image_root)
            return
        reports_dir = CLINICAL_DATA_DIR / "synthetic"
        splits = build_synthetic_splits(image_splits, reports_dir)
        logger.info("Generated synthetic reports in %s", reports_dir)
    elif args.data.suffix.lower() == ".csv":
        splits = load_reports_csv(args.data, return_features=True)
    else:
        splits = load_reports_from_folders(args.data)

    if "train" not in splits or not splits["train"][0]:
        logger.error("No training reports found.")
        return

    train_texts, train_labels = splits["train"][:2]
    train_features = splits["train"][2] if len(splits["train"]) > 2 else None
    
    val_texts = splits.get("val", (None, None))[:2][0]
    val_labels = splits.get("val", (None, None))[:2][1]
    val_features = splits.get("val", ([], [], []))[2] if len(splits.get("val", [])) > 2 else None
    
    test_texts = splits.get("test", (None, None))[:2][0]
    test_labels = splits.get("test", (None, None))[:2][1]
    test_features = splits.get("test", ([], [], []))[2] if len(splits.get("test", [])) > 2 else None
    
    logger.info("Train: %d reports | Val: %d | Test: %d",
                len(train_texts), len(val_texts or []), len(test_texts or []))

    if MLFLOW_ENABLED:
        tracker.start_run(run_name=f"clinical_{time.strftime('%Y%m%d_%H%M%S')}",
                          tags={"phase": "clinical"})
        tracker.log_params({"model": "Bio_ClinicalBERT", "epochs": args.epochs,
                            "batch_size": args.batch_size, "lr": args.lr,
                            "freeze": args.freeze, "num_classes": args.num_classes,
                            "ordinal": args.ordinal})

    train_ds = ClinicalTextDataset(train_texts, train_labels, features=train_features, augment=True, max_length=CLINICAL_MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = test_loader = None
    if val_texts:
        val_loader = DataLoader(ClinicalTextDataset(val_texts, val_labels, features=val_features, augment=False, max_length=CLINICAL_MAX_LENGTH),
                                batch_size=args.batch_size, num_workers=0)
    if test_texts:
        test_loader = DataLoader(ClinicalTextDataset(test_texts, test_labels, features=test_features, augment=False, max_length=CLINICAL_MAX_LENGTH),
                                 batch_size=args.batch_size, num_workers=0)

    model = ClinicalTextModel(num_classes=args.num_classes, freeze_encoder=args.freeze, ordinal=args.ordinal).to(device)
    
    alpha = torch.tensor(class_weights(train_labels, num_classes=args.num_classes),
                         dtype=torch.float).to(device)
                         
    if args.ordinal:
        from kneevision.training.losses import OrdinalLoss
        criterion = OrdinalLoss(num_classes=args.num_classes, alpha=alpha)
    else:
        criterion = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best_kappa = -1.0
    best_path = MODELS_DIR / "best_clinical.pt"
    MODELS_DIR.mkdir(exist_ok=True)
    no_improve = 0  # early stopping counter
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss = criterion(model(input_ids, attention_mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        log_line = f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f}"

        if val_loader is not None:
            val_loss, val_acc, val_kappa, _, _ = validate(model, val_loader, criterion, device, is_ordinal=args.ordinal)
            log_line += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Kappa: {val_kappa:.4f}"
            if MLFLOW_ENABLED:
                tracker.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                                     "val_acc": val_acc, "val_kappa": val_kappa}, step=epoch)
            scheduler.step(val_kappa)
            if val_kappa > best_kappa:
                best_kappa = val_kappa
                no_improve = 0
                torch.save(model.state_dict(), best_path)
                log_line += "  -> saved"
            else:
                no_improve += 1
                log_line += f"  (no improve {no_improve}/{args.patience})"
        else:
            if MLFLOW_ENABLED:
                tracker.log_metrics({"train_loss": train_loss}, step=epoch)

        logger.info(log_line)

        if val_loader is not None and no_improve >= args.patience:
            logger.info("Early stopping triggered after %d epochs with no improvement.", no_improve)
            break

    logger.info("Best val kappa: %.4f -> %s", best_kappa, best_path)

    if test_loader is not None:
        test_loss, test_acc, test_kappa, preds, labels = validate(model, test_loader, criterion, device, is_ordinal=args.ordinal)
        logger.info("=" * 60)
        logger.info("TEST REPORT")
        logger.info("=" * 60)
        logger.info("Test Accuracy: %.4f | Test Kappa: %.4f", test_acc, test_kappa)
        logger.info("\n%s", classification_report(labels, preds, digits=4))
        if MLFLOW_ENABLED:
            tracker.log_metrics({"test_accuracy": test_acc, "test_kappa": test_kappa})
            tracker.end_run()

    if not val_loader:
        torch.save(model.state_dict(), best_path)
        logger.info("Saved model to %s", best_path)


if __name__ == "__main__":
    main()
