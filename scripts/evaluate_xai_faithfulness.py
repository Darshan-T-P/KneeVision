"""Quantitative XAI faithfulness: deletion / insertion curves for Grad-CAM and Score-CAM.

For each image the predicted grade's probability is tracked while 16x16 patches are
removed (deletion; lower AUC = more faithful) or revealed (insertion; higher AUC =
more faithful) in order of decreasing saliency. A random-saliency baseline gives the
reference. Runs on the VALIDATION split only, so the frozen test set is untouched.

Usage:
    uv run python scripts/evaluate_xai_faithfulness.py --n 100 --scorecam-n 20
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from PIL import Image

from kneevision.config.settings import MODELS_DIR, PROJECT_ROOT
from kneevision.data.transforms import val_transform
from kneevision.models.image_model import load_trained_model
from kneevision.training.losses import ordinal_to_class, ordinal_to_probs
from kneevision.utils.helpers import get_device, set_seed
from kneevision.xai.gradcam import GradCAM
from kneevision.xai.lime import explain as lime_explain
from kneevision.xai.scorecam import ScoreCAM

PATCH = 16
GRID = 224 // PATCH


@torch.inference_mode()
def _prob(model, x, cls):
    logits = model(x)
    if getattr(model, "ordinal", False):
        p = ordinal_to_probs(logits)
    else:
        p = torch.softmax(logits, dim=1)
    return p[:, cls]


def _patch_order(cam: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    pooled = cam.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3)).ravel()
    return np.argsort(-pooled)


def _curves(model, x, cls, order):
    """Return (deletion, insertion) probability curves over GRID*GRID+1 steps."""
    n = GRID * GRID
    base = torch.zeros_like(x)  # 0 == dataset mean after normalisation
    masks = torch.zeros(n + 1, 1, 224, 224, device=x.device)
    flat = torch.zeros(GRID, GRID, device=x.device).flatten()
    for k, idx in enumerate(order, start=1):
        flat[idx] = 1.0
        masks[k, 0] = flat.view(GRID, GRID).repeat_interleave(PATCH, 0).repeat_interleave(PATCH, 1)
    ins_in = base + masks * (x - base)
    del_in = x - masks * (x - base)
    ins = torch.cat([_prob(model, ins_in[i:i + 64], cls) for i in range(0, n + 1, 64)])
    dele = torch.cat([_prob(model, del_in[i:i + 64], cls) for i in range(0, n + 1, 64)])
    return dele.cpu().numpy(), ins.cpu().numpy()


def _auc(y):
    return float(np.trapezoid(y, dx=1.0 / (len(y) - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(MODELS_DIR / "best_densenet121_ordinal_ordinal_soft_mixup_a04.pt"))
    ap.add_argument("--n", type=int, default=100, help="images for Grad-CAM/random (val split, stratified)")
    ap.add_argument("--scorecam-n", type=int, default=20, help="subset for the slow Score-CAM")
    ap.add_argument("--lime-n", type=int, default=0, help="subset for LIME (500 perturbations each; slow)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default=str(PROJECT_ROOT / "reports" / "xai_faithfulness.json"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    model = load_trained_model(args.checkpoint, device, num_classes=5).to(device).eval()

    rng = np.random.default_rng(args.seed)
    paths = []
    per = max(1, args.n // 5)
    for k in range(5):
        files = sorted((PROJECT_ROOT / "data/raw/val" / str(k)).glob("*.png"))
        paths += [files[i] for i in rng.choice(len(files), min(per, len(files)), replace=False)]

    gc = GradCAM(model)
    sc = ScoreCAM(model)
    res = {"gradcam": [], "scorecam": [], "lime": [], "random": []}
    for i, p in enumerate(paths):
        x = val_transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(x)
        cls = int(ordinal_to_class(logits).item()) if model.ordinal else int(logits.argmax(1).item())
        cls = min(cls, 4)
        methods = {
            "gradcam": gc.generate(x.clone(), class_idx=cls),
            "random": rng.random((224, 224)),
        }
        if i % max(1, len(paths) // max(1, args.scorecam_n)) == 0 and len(res["scorecam"]) < args.scorecam_n:
            methods["scorecam"] = sc.generate(x.clone(), class_idx=cls)
        if i % max(1, len(paths) // max(1, args.lime_n)) == 0 and len(res["lime"]) < args.lime_n:
            _, _, imp, _ = lime_explain(model, Image.open(p).convert("RGB"), val_transform, device)
            methods["lime"] = np.asarray(imp)
        for name, cam in methods.items():
            d, ins = _curves(model, x, cls, _patch_order(cam))
            res[name].append((_auc(d), _auc(ins)))
        print(f"[{i + 1}/{len(paths)}] {p.name}", flush=True)

    out = {"split": "val", "checkpoint": Path(args.checkpoint).name, "patch": PATCH, "seed": args.seed, "methods": {}}
    for name, v in res.items():
        if not v:
            continue
        a = np.array(v)
        out["methods"][name] = {
            "n": len(a),
            "deletion_auc_mean": float(a[:, 0].mean()), "deletion_auc_sd": float(a[:, 0].std(ddof=1)),
            "insertion_auc_mean": float(a[:, 1].mean()), "insertion_auc_sd": float(a[:, 1].std(ddof=1)),
        }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["methods"], indent=2))


if __name__ == "__main__":
    main()
