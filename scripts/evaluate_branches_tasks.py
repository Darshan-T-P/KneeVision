"""Per-branch predictions and task-variant accuracies (5-class, 3-class, binary) for clinical, image and fusion models.

Re-runs the SAME frozen models on the SAME test set as scripts/evaluate_fusion.py (argmax of class probabilities),
saves per-sample predictions and confusion matrices, and derives binary (KL0-1 vs KL2-4) and 3-class
(KL0-1 / KL2 / KL3-4) accuracy. Evaluation only; no training, no tuning.

Usage: uv run python scripts/evaluate_branches_tasks.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from kneevision.clinical.model import load_trained_clinical_model
from kneevision.config.settings import OAI_DATA_DIR, RAW_DATA_DIR
from kneevision.data.prepare import prepare_from_folders
from kneevision.data.transforms import val_transform
from kneevision.fusion import MultimodalDataset, load_trained_fusion_model
from kneevision.models.image_model import load_trained_model as load_image_model
from kneevision.training.losses import ordinal_to_probs
from kneevision.utils.helpers import get_device

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "branches_tasks"
OUT.mkdir(parents=True, exist_ok=True)


def features(csv_path):
    d = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid, side = row.get("id", ""), row.get("side", "")
            d[(pid, side)] = row
            if pid and not side:
                d[(pid, "")] = row
    return d


def main():
    device = get_device()
    paths, labels = prepare_from_folders(RAW_DATA_DIR)["test"]
    ds = MultimodalDataset(image_paths=paths, labels=labels, features_dict=features(OAI_DATA_DIR / "processed" / "oai_clinical.csv"),
                           augment_text=False, transform=val_transform, allow_fallback_text=True)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    fusion = load_trained_fusion_model(ROOT / "models/best_fusion.pt", device, num_classes=5).eval()
    img = load_image_model(ROOT / "models/best_densenet121.pt", device, num_classes=5).eval()
    txt = load_trained_clinical_model(ROOT / "models/best_clinical.pt", device, num_classes=5).eval()

    y, P = [], {"fusion": [], "image": [], "clinical": []}
    with torch.inference_mode():
        for b in tqdm(dl, desc="test"):
            im, ids, am = b["image"].to(device), b["input_ids"].to(device), b["attention_mask"].to(device)
            y.extend(b["label"].numpy().tolist())
            P["fusion"].append(fusion.predict_probs(im, ids, am).cpu().numpy())
            lg = img(im)
            P["image"].append((ordinal_to_probs(lg) if getattr(img, "ordinal", False) else torch.softmax(lg, 1)).cpu().numpy())
            lt = txt(ids, am)
            P["clinical"].append((ordinal_to_probs(lt) if txt.ordinal else torch.softmax(lt, 1)).cpu().numpy())
    y = np.array(y)
    res = {}
    for name, chunks in P.items():
        pr = np.concatenate(chunks)
        p = pr.argmax(1)
        cm = np.zeros((5, 5), int); np.add.at(cm, (y, p), 1)
        grp3 = lambda a: np.where(a <= 1, 0, np.where(a == 2, 1, 2))
        res[name] = {
            "acc5": float((y == p).mean()), "qwk": float(cohen_kappa_score(y, p, weights="quadratic")),
            "acc_binary": float(((y >= 2) == (p >= 2)).mean()), "acc_3class_01_2_34": float((grp3(y) == grp3(p)).mean()),
            "sens_binary": float((p[y >= 2] >= 2).mean()), "spec_binary": float((p[y < 2] < 2).mean()),
            "n": int(len(y)), "confusion_matrix": cm.tolist(),
        }
        with open(OUT / f"{name}_predictions.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["image_path", "true_label", "pred_label"] + [f"prob_KL{k}" for k in range(5)])
            for pth, t, pp, pb in zip(paths, y, p, pr):
                w.writerow([pth, int(t), int(pp)] + [float(v) for v in pb])
        print(name, {k: round(v, 4) for k, v in res[name].items() if isinstance(v, float)})
    (OUT / "branch_task_results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
