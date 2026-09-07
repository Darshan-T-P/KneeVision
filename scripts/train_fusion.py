import argparse
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report

from kneevision.fusion.model import MultimodalFusionModel
from kneevision.fusion.dataset import MultimodalDataset
from kneevision.models.image_model import load_trained_model as load_image_model
from kneevision.clinical.model import ClinicalTextModel
from kneevision.data.prepare import prepare_from_folders
from kneevision.data.transforms import build_train_transform, build_val_transform
from kneevision.training.losses import OrdinalLoss, ordinal_to_class
from kneevision.utils.logging import setup_logger

logger = setup_logger("train_fusion")

def load_features_dict(csv_path: Path):
    features_dict = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid, side = row["id"], row["side"]
            features_dict[(pid, side)] = row
    return features_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-model", type=str, default="models/best_densenet121.pt")
    parser.add_argument("--text-model", type=str, default="models/best_clinical.pt")
    parser.add_argument("--image-data", type=str, default="data/raw")
    parser.add_argument("--text-csv", type=str, default="data/oai/processed/oai_clinical.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ordinal", action="store_true", default=True) # Always use ordinal for KL
    parser.add_argument("--unfreeze", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info("Loading models...")
    img_model = load_image_model(args.image_model, device, num_classes=5)
    
    # Load text model state dict if it's a pt file
    txt_model = ClinicalTextModel(ordinal=args.ordinal).to(device)
    if Path(args.text_model).exists():
        data = torch.load(args.text_model, map_location=device, weights_only=False)
        if isinstance(data, dict) and "model_state_dict" in data:
             state = data["model_state_dict"]
        else:
             state = data
        txt_model.load_state_dict(state)

    model = MultimodalFusionModel(img_model, txt_model, ordinal=args.ordinal, freeze_encoders=not args.unfreeze).to(device)

    logger.info("Loading data...")
    features_dict = load_features_dict(Path(args.text_csv))
    splits = prepare_from_folders(Path(args.image_data))
    
    train_tf = build_train_transform(size=224)
    val_tf = build_val_transform(size=224)
    
    datasets = {}
    datasets["train"] = MultimodalDataset(splits["train"][0], splits["train"][1], features_dict, augment_text=True, transform=train_tf)
    datasets["val"] = MultimodalDataset(splits["val"][0], splits["val"][1], features_dict, augment_text=False, transform=val_tf)
    datasets["test"] = MultimodalDataset(splits["test"][0], splits["test"][1], features_dict, augment_text=False, transform=val_tf)
    
    logger.info(f"Train: {len(datasets['train'])} | Val: {len(datasets['val'])} | Test: {len(datasets['test'])}")
    
    loaders = {
        split: DataLoader(ds, batch_size=args.batch_size, shuffle=(split=="train"), num_workers=4, pin_memory=True)
        for split, ds in datasets.items()
    }

    # Ordinal Loss
    # We can infer alpha weights from training distribution
    from kneevision.data.prepare import class_weights
    train_labels = [datasets["train"].labels[i] for i in range(len(datasets["train"]))]
    alpha_weights = torch.tensor(class_weights(train_labels)).to(device)
    criterion = OrdinalLoss(num_classes=5, alpha=alpha_weights)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    best_kappa = -1.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            imgs = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(imgs, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(loaders["train"])
        
        # Eval
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loaders["val"]:
                imgs = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                logits = model(imgs, input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = ordinal_to_class(logits)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
        val_loss /= len(loaders["val"])
        val_acc = accuracy_score(all_labels, all_preds)
        val_kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
        
        status = f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Kappa: {val_kappa:.4f}"
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), "models/best_fusion.pt")
            logger.info(f"Epoch {epoch:2d} | {status}  -> saved")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"Epoch {epoch:2d} | {status}  (no improve {patience_counter}/3)")
            if patience_counter >= 3:
                logger.info(f"Early stopping triggered after {patience_counter} epochs with no improvement.")
                break
                
    # Test
    logger.info("TESTING")
    model.load_state_dict(torch.load("models/best_fusion.pt", weights_only=True))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loaders["test"]:
            imgs = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(imgs, input_ids, attention_mask)
            preds = ordinal_to_class(logits)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
    test_acc = accuracy_score(all_labels, all_preds)
    test_kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    logger.info(f"Test Accuracy: {test_acc:.4f} | Test Kappa: {test_kappa:.4f}")
    logger.info("\n" + classification_report(all_labels, all_preds))

if __name__ == "__main__":
    main()
