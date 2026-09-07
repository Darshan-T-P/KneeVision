"""Evaluate Multimodal Fusion Model on Knee OA severity grading test set.

Compares:
1. Image-only CNN (DenseNet121)
2. Clinical-text NLP (BioClinicalBERT)
3. Multimodal Late Fusion Network (DenseNet121 + BioClinicalBERT)

Usage:
    uv run python scripts/evaluate_fusion.py
    uv run python scripts/evaluate_fusion.py --fusion-model models/best_fusion.pt
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kneevision.config.settings import (
    RAW_DATA_DIR,
    BATCH_SIZE,
    MLFLOW_ENABLED,
    OAI_DATA_DIR,
)
from kneevision.fusion import load_trained_fusion_model, MultimodalDataset
from kneevision.models.image_model import load_trained_model as load_image_model
from kneevision.clinical.model import ClinicalTextModel
from kneevision.data.prepare import prepare_from_folders
from kneevision.data.transforms import val_transform
from kneevision.evaluation.report import (
    compute_metrics,
    classification_report_text,
    confusion_matrix_plot,
    html_report,
)
from kneevision.training.losses import ordinal_to_probs
from kneevision.utils.helpers import get_device
from kneevision.utils.logging import setup_logger

logger = setup_logger("evaluate_fusion")


def load_features_dict(csv_path: Path) -> dict:
    features_dict = {}
    if not csv_path.exists():
        logger.warning(f"Clinical CSV not found at {csv_path}")
        return features_dict

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid, side = row.get("id", ""), row.get("side", "")
            features_dict[(pid, side)] = row
            if pid and not side:
                features_dict[(pid, "")] = row
    return features_dict


def plot_modality_comparison(comparison_data: dict[str, dict[str, float]], out_path: Path) -> plt.Figure:
    """Create a publication-quality comparative bar chart across modalities."""
    metrics_to_plot = ["accuracy", "kappa", "macro_f1", "weighted_f1"]
    metric_labels = ["Accuracy", "Quadratic Kappa", "Macro F1", "Weighted F1"]
    modalities = list(comparison_data.keys())

    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(modalities)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#0284c7", "#7c3aed", "#0d9488", "#f59e0b"]

    for i, mod in enumerate(modalities):
        values = [comparison_data[mod].get(m, 0.0) for m in metrics_to_plot]
        offset = (i - len(modalities) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=mod, color=colors[i % len(colors)], alpha=0.9, edgecolor="black", linewidth=0.8)
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 0.012,
                f"{yval:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="semibold",
            )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Modality Comparison on Knee OA Test Set (KL 0-4)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=10.5)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Evaluate KneeVision++ Multimodal Fusion Model")
    parser.add_argument("--fusion-model", type=str, default="models/best_fusion.pt", help="Path to fusion checkpoint")
    parser.add_argument("--image-model", type=str, default="models/best_densenet121.pt", help="Baseline image model")
    parser.add_argument("--text-model", type=str, default="models/best_clinical.pt", help="Baseline clinical text model")
    parser.add_argument("--image-data", type=str, default=str(RAW_DATA_DIR), help="Path to raw image split directory")
    parser.add_argument("--text-csv", type=str, default=str(OAI_DATA_DIR / "processed" / "oai_clinical.csv"), help="Clinical CSV")
    parser.add_argument("--out-dir", type=str, default="reports/evaluate_fusion", help="Output directory for reports")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    logger.info("Loading test dataset splits...")
    splits = prepare_from_folders(Path(args.image_data))
    test_paths, test_labels = splits["test"]
    features_dict = load_features_dict(Path(args.text_csv))

    test_ds = MultimodalDataset(
        image_paths=test_paths,
        labels=test_labels,
        features_dict=features_dict,
        augment_text=False,
        transform=val_transform,
        allow_fallback_text=True,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    logger.info(f"Total test samples: {len(test_ds)}")

    # 2. Evaluate Multimodal Fusion Model
    fusion_path = Path(args.fusion_model)
    if not fusion_path.exists():
        logger.error(f"Fusion model not found at {fusion_path}. Please train first with scripts/train_fusion.py")
        sys.exit(1)

    logger.info(f"Loading fusion model from {fusion_path}...")
    fusion_model = load_trained_fusion_model(fusion_path, device, num_classes=5)
    fusion_model.eval()

    all_labels, fusion_preds, fusion_probs = [], [], []
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Evaluating Fusion"):
            imgs = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            probs = fusion_model.predict_probs(imgs, input_ids, attention_mask)
            preds = probs.argmax(dim=-1)

            all_labels.extend(labels.numpy())
            fusion_preds.extend(preds.cpu().numpy())
            fusion_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    fusion_preds = np.array(fusion_preds)
    fusion_probs = np.array(fusion_probs)

    fusion_metrics = compute_metrics(all_labels, fusion_preds, fusion_probs)
    logger.info("=== MULTIMODAL FUSION METRICS ===")
    logger.info(f"Accuracy:        {fusion_metrics['accuracy']:.4f}")
    logger.info(f"Quadratic Kappa: {fusion_metrics['kappa']:.4f}")
    logger.info(f"Macro F1:        {fusion_metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1:     {fusion_metrics['weighted_f1']:.4f}")

    comparison = {"Multimodal Fusion": fusion_metrics}

    # 3. Evaluate Image-Only Baseline
    img_model_path = Path(args.image_model)
    if img_model_path.exists():
        try:
            logger.info(f"Evaluating Image Baseline ({img_model_path.name})...")
            img_model = load_image_model(img_model_path, device, num_classes=5)
            img_model.eval()

            img_preds, img_probs = [], []
            with torch.inference_mode():
                for batch in tqdm(test_loader, desc="Evaluating Image Baseline"):
                    imgs = batch["image"].to(device)
                    logits = img_model(imgs)
                    if getattr(img_model, "ordinal", False):
                        p = ordinal_to_probs(logits)
                    else:
                        p = torch.softmax(logits, dim=1)
                    img_preds.extend(p.argmax(dim=1).cpu().numpy())
                    img_probs.extend(p.cpu().numpy())

            img_metrics = compute_metrics(all_labels, np.array(img_preds), np.array(img_probs))
            comparison[f"Image ({img_model_path.stem})"] = img_metrics
            logger.info(f"Image Baseline Kappa: {img_metrics['kappa']:.4f} | Acc: {img_metrics['accuracy']:.4f}")
        except Exception as e:
            logger.warning(f"Could not evaluate image baseline: {e}")

    # 4. Evaluate Clinical Text Baseline
    txt_model_path = Path(args.text_model)
    if txt_model_path.exists():
        try:
            logger.info(f"Evaluating Clinical Text Baseline ({txt_model_path.name})...")
            state = torch.load(txt_model_path, map_location=device, weights_only=False)
            is_ordinal = False
            if isinstance(state, dict):
                st_dict = state.get("model_state_dict", state)
                if "classifier.weight" in st_dict and st_dict["classifier.weight"].shape[0] == 4:
                    is_ordinal = True
            txt_model = ClinicalTextModel(num_classes=5, ordinal=is_ordinal).to(device)
            txt_model.load_state_dict(st_dict if isinstance(state, dict) else state)
            txt_model.eval()

            txt_preds, txt_probs = [], []
            with torch.inference_mode():
                for batch in tqdm(test_loader, desc="Evaluating Text Baseline"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    logits = txt_model(input_ids, attention_mask)
                    if is_ordinal:
                        p = ordinal_to_probs(logits)
                    else:
                        p = torch.softmax(logits, dim=1)
                    txt_preds.extend(p.argmax(dim=1).cpu().numpy())
                    txt_probs.extend(p.cpu().numpy())

            txt_metrics = compute_metrics(all_labels, np.array(txt_preds), np.array(txt_probs))
            comparison["Clinical Text (BioClinicalBERT)"] = txt_metrics
            logger.info(f"Text Baseline Kappa: {txt_metrics['kappa']:.4f} | Acc: {txt_metrics['accuracy']:.4f}")
        except Exception as e:
            logger.warning(f"Could not evaluate text baseline: {e}")

    # 5. Generate Artifacts & Reports
    cm_path = out_dir / "confusion_matrix.png"
    cm_fig = confusion_matrix_plot(all_labels, fusion_preds, out_path=cm_path)

    comp_path = out_dir / "modality_comparison.png"
    plot_modality_comparison(comparison, comp_path)

    rep_txt = classification_report_text(all_labels, fusion_preds)
    (out_dir / "classification_report.txt").write_text(rep_txt)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(comparison, f, indent=2)

    extra_notes = "<h3>Modality Comparison</h3><p>Evaluated on 1,656 test samples using 5-grade Kellgren-Lawrence severity scale.</p>"
    html_content = html_report(fusion_metrics, cm_fig, rep_txt, extra_notes=extra_notes)
    (out_dir / "report.html").write_text(html_content)
    logger.info(f"Saved evaluation artifacts to {out_dir}/")

    # 6. Log to MLflow if enabled
    if MLFLOW_ENABLED and not args.no_mlflow:
        try:
            from kneevision.utils.tracking import MLflowTracker
            tracker = MLflowTracker()
            run_name = f"eval_fusion_{time.strftime('%Y%m%d_%H%M%S')}"
            tracker.start_run(run_name=run_name, tags={"phase": "phase_5_multimodal_fusion"})
            tracker.log_params({
                "fusion_model": str(fusion_path),
                "test_samples": len(test_ds),
                "num_classes": 5,
                "batch_size": args.batch_size,
            })
            tracker.log_metrics(fusion_metrics)
            tracker.log_artifact(str(cm_path))
            tracker.log_artifact(str(comp_path))
            tracker.log_artifact(str(out_dir / "classification_report.txt"))
            tracker.log_artifact(str(out_dir / "report.html"))
            tracker.end_run()
            logger.info("Logged evaluation metrics and artifacts to MLflow.")
        except Exception as e:
            logger.warning(f"MLflow logging skipped: {e}")

    print("\n" + "=" * 60)
    print("KNEEVISION++ MULTIMODAL FUSION TEST RESULTS")
    print("=" * 60)
    print(rep_txt)
    print("=" * 60)


if __name__ == "__main__":
    main()
