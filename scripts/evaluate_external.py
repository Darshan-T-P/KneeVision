"""External-validation evaluation on a folder of knee X-rays NOT drawn from the training set.

Expects class sub-folders whose names contain the KL grade digit (0-4), e.g.
``0/``, ``KL1/``, ``Grade_2/``, ``3_moderate/``. Folder-to-grade mapping is printed
before inference; override it with ``--label-map "normal=0,doubtful=1,..."``.
Roboflow "Folder Structure" exports with a split level (train/valid/test) are
handled: pass ``--root`` at the export root and all splits are pooled, or at a
single split.

Evaluation only: single non-TTA forward pass, no training, no tuning.

Usage:
    uv run python scripts/evaluate_external.py --root data/external/roboflow \\
        --checkpoint models/best_densenet121_ordinal_ordinal_soft_mixup_a04.pt \\
        --out-json reports/external_eval.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from kneevision.config.settings import BATCH_SIZE, PROJECT_ROOT
from kneevision.data.dataset import KneeXRayDataset
from kneevision.data.preprocess import PREPROCESS_CHOICES
from kneevision.data.transforms import build_val_transform, val_transform
from kneevision.evaluation.report import (
    classification_report_text,
    compute_confusion_matrix,
    compute_metrics,
)
from kneevision.models.image_model import load_trained_model
from kneevision.training.losses import ordinal_to_class, ordinal_to_probs
from kneevision.utils.helpers import get_device

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_densenet121_ordinal_ordinal_soft_mixup_a04.pt"
CLASS_NAMES = ["KL0", "KL1", "KL2", "KL3", "KL4"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_label_map(spec: str | None) -> dict[str, int]:
    if not spec:
        return {}
    out = {}
    for item in spec.split(","):
        name, grade = item.split("=")
        out[name.strip().lower()] = int(grade)
    return out


def folder_to_grade(name: str, explicit: dict[str, int]) -> int | None:
    key = name.lower()
    if key in explicit:
        return explicit[key]
    digits = re.findall(r"\d", name)
    if len(digits) == 1 and 0 <= int(digits[0]) <= 4:
        return int(digits[0])
    return None


def collect(root: Path, explicit: dict[str, int]) -> tuple[list[str], list[int], dict[str, int]]:
    paths, labels, mapping = [], [], {}
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        grade = folder_to_grade(img.parent.name, explicit)
        if grade is None:
            mapping.setdefault(img.parent.name, -1)
            continue
        mapping[img.parent.name] = grade
        paths.append(str(img))
        labels.append(grade)
    return paths, labels, mapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="Folder containing class sub-folders of images")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--label-map", type=str, default=None, help='e.g. "normal=0,doubtful=1,mild=2,moderate=3,severe=4"')
    ap.add_argument("--allow-non-ordinal", action="store_true")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--preprocess", choices=PREPROCESS_CHOICES, default="none",
                    help="must match the preprocessing the checkpoint was trained with")
    ap.add_argument("--pad-square", action="store_true",
                    help="letterbox each image to a square (median-intensity padding) before the usual resize, "
                         "instead of stretching non-square images")
    ap.add_argument("--save-predictions", type=Path, default=None, help="write per-image predictions + probabilities (CSV)")
    ap.add_argument("--dry-run", action="store_true", help="Only print the folder->grade mapping and counts")
    args = ap.parse_args()

    paths, labels, mapping = collect(args.root, parse_label_map(args.label_map))
    print("Folder -> KL grade mapping (-1 = unmapped, skipped):")
    for folder, grade in sorted(mapping.items(), key=lambda kv: kv[1]):
        print(f"  {folder!r:30} -> {grade}")
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=5) if labels else np.zeros(5, int)
    print(f"Images: {len(paths)}  per-grade: {dict(zip(CLASS_NAMES, counts.tolist()))}")
    if not paths:
        raise SystemExit("No labelled images found. Check --root or pass --label-map.")
    if args.dry_run:
        return

    device = get_device()
    model = load_trained_model(args.checkpoint.resolve(), device, num_classes=5)
    is_ordinal = getattr(model, "ordinal", False)
    if not is_ordinal and not args.allow_non_ordinal:
        raise SystemExit("Checkpoint is not ordinal; pass --allow-non-ordinal to evaluate it.")
    model.eval()

    transform = build_val_transform(preprocess=args.preprocess) if args.preprocess != "none" else val_transform
    if args.pad_square:
        from PIL import Image as _Image
        from torchvision import transforms as _T

        def _letterbox(img):
            w, h = img.size
            side = max(w, h)
            fill = tuple(int(v) for v in np.median(np.asarray(img).reshape(-1, 3), axis=0))
            canvas = _Image.new("RGB", (side, side), fill)
            canvas.paste(img, ((side - w) // 2, (side - h) // 2))
            return canvas

        transform = _T.Compose([_T.Lambda(_letterbox), transform])
    loader = DataLoader(KneeXRayDataset(paths, labels, transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    preds, probs = [], []
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc="External inference"):
            logits = model(images.to(device))
            if is_ordinal:
                preds.extend(ordinal_to_class(logits).cpu().tolist())
                probs.extend(ordinal_to_probs(logits).cpu().tolist())
            else:
                p = F.softmax(logits, dim=1)
                preds.extend(p.argmax(dim=1).cpu().tolist())
                probs.extend(p.cpu().tolist())

    y, yhat, pr = np.asarray(labels), np.asarray(preds), np.asarray(probs)
    metrics = compute_metrics(y, yhat, probs=pr, num_classes=5)
    cm = compute_confusion_matrix(y, yhat, num_classes=5)
    print(classification_report_text(y, yhat, class_names=CLASS_NAMES, num_classes=5))
    print("Confusion matrix (rows=true, cols=pred):\n", cm)
    print({k: round(float(v), 4) for k, v in metrics.items() if k == "accuracy" or "kappa" in k or k == "macro_f1"})

    if args.save_predictions:
        import csv
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_predictions, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["image_path", "true_label", "pred_label"] + [f"prob_KL{k}" for k in range(5)])
            for pth, t, pr_, pb in zip(paths, y, yhat, pr):
                w.writerow([pth, int(t), int(pr_)] + [float(v) for v in pb])

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            "root": str(args.root), "pad_square": bool(args.pad_square), "preprocess": args.preprocess, "checkpoint": str(args.checkpoint), "n": len(paths),
            "folder_mapping": mapping, "metrics": {k: float(v) for k, v in metrics.items()},
            "confusion_matrix": np.asarray(cm).tolist(),
        }, indent=2))
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
