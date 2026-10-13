"""Tier-A external generalization study: zero-shot vs cheap adaptation, on the fixed dev/test split.

Methods (all pre-specified; every one is reported, none is selected on the test part):
  zero_shot        : model as trained.
  adabn            : re-estimate BatchNorm statistics on UNLABELLED dev images (unsupervised domain
                     adaptation; no weights change). Not applicable to LayerNorm-only nets (Swin, ConvNeXt).
  recal_n{N}       : shift the 4 CORAL threshold biases to maximize QWK on N LABELLED dev images
                     (few-shot local calibration; ordinal models only). N in {50, 100, all}.
  adabn_recal_n{N} : both.
Everything is evaluated on the held-out 70% TEST part only. Recalibration with N<all is averaged over 5
random draws of N dev labels. Image-level bootstrap CI (no patient IDs available).

Usage: uv run python scripts/evaluate_external_adapt.py
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from kneevision.config.settings import PROJECT_ROOT
from kneevision.data.dataset import KneeXRayDataset
from kneevision.data.transforms import build_val_transform, val_transform
from kneevision.models.image_model import load_trained_model
from kneevision.training.losses import ordinal_to_probs
from kneevision.utils.helpers import get_device, set_seed

M = PROJECT_ROOT / "models"
A = "best_densenet121_ordinal_ordinal_soft_mixup_a04"
MT = "best_densenet121_multitask_ordinal_soft_mixup_a04"
MODELS = {
    "plain CORAL (DenseNet121)": [A, A + "_seed7", A + "_seed123", A + "_seed2024"],
    "multitask CORAL (DenseNet121)": [MT, MT + "_seed7", MT + "_seed2024"],
    "Focal baseline (DenseNet121)": ["best_densenet121_focal_nomixup"],
    "ConvNeXt-Tiny CORAL": ["best_convnext_tiny_ordinal_ordinal_soft_mixup_a04"],
    "EfficientNet-B4 CORAL": ["best_efficientnet-b4_ordinal_ordinal_soft_mixup_a04"],
    "Swin-T CORAL": ["best_swin_t_ordinal_ordinal_soft_mixup_a04"],
}
TIER_B = {  # 40-epoch schedule, soft CORAL + MixUp 0.4; base = default pipeline, robust = domain-robust aug + per-image standardization
    "short-schedule baseline (DenseNet121)": [f"best_densenet121_ordinal_short40_base_seed{s}" for s in (42, 7)],
    "domain-robust training (DenseNet121)": [f"best_densenet121_ordinal_short40_robust_seed{s}" for s in (42, 7)],
}
GRID = np.linspace(-4, 4, 33)


def loader(items, device, robust=False):
    tf = build_val_transform(robust=True) if robust else val_transform
    paths = [str(PROJECT_ROOT / p) for p, _ in items]
    labels = [g for _, g in items]
    return DataLoader(KneeXRayDataset(paths, labels, tf), batch_size=32, shuffle=False, num_workers=4), np.array(labels)


@torch.inference_mode()
def logits_of(model, dl, device):
    model.eval()
    return torch.cat([model(x.to(device)).float().cpu() for x, _ in dl])


@torch.inference_mode()
def adabn(model, dl_unlabelled, device):
    m = copy.deepcopy(model).to(device)
    bns = [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return None
    m.eval()
    for b in bns:
        b.reset_running_stats()
        b.momentum = None  # cumulative average over all adaptation images
        b.train()
    for x, _ in dl_unlabelled:
        m(x.to(device))
    return m.eval()


def decode(z, b):
    return (torch.sigmoid(z + b) >= 0.5).sum(1).numpy()


def fit_shifts(z, y):
    b = torch.zeros(z.shape[1])
    for _ in range(3):
        for k in range(z.shape[1]):
            best, best_q = b[k].item(), -9
            for g in GRID:
                bb = b.clone(); bb[k] = g
                q = cohen_kappa_score(y, decode(z, bb), weights="quadratic")
                if q > best_q + 1e-9 or (abs(q - best_q) <= 1e-9 and abs(g) < abs(best)):
                    best, best_q = float(g), q
            b[k] = best
    return b


def metrics(y, p, probs=None, ci=False):
    cm = np.zeros((5, 5), int); np.add.at(cm, (y, p), 1)
    yb, pb = y >= 2, p >= 2
    m = {"qwk": cohen_kappa_score(y, p, weights="quadratic"), "acc": float((y == p).mean()), "mae": float(np.abs(y - p).mean()),
         "within1": float((np.abs(y - p) <= 1).mean()), "bin_acc": float((yb == pb).mean()),
         "sens": float(pb[yb].mean()), "spec": float((~pb[~yb]).mean()),
         "recall": [float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(5)], "pred_counts": np.bincount(p, minlength=5).tolist()}
    prec = np.diag(cm) / np.maximum(cm.sum(0), 1); rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    m["macro_f1"] = float(np.mean(np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)))
    if probs is not None:
        m["bin_auc"] = float(roc_auc_score(yb, probs[:, 2:].sum(1)))
    if ci:
        rng = np.random.default_rng(42); n = len(y); q = []
        for _ in range(1000):
            i = rng.integers(0, n, n); q.append(cohen_kappa_score(y[i], p[i], weights="quadratic"))
        m["qwk_ci"] = [float(np.percentile(q, 2.5)), float(np.percentile(q, 97.5))]
    return m


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-b", action="store_true", help="evaluate the Tier-B models instead of the Tier-A set")
    args = ap.parse_args()
    models = TIER_B if args.tier_b else MODELS
    set_seed(42)
    device = get_device()
    split = json.loads((PROJECT_ROOT / "reports" / "external_gornale_split.json").read_text())
    loaders = {r: (loader(split["dev"], device, r), loader(split["test"], device, r)) for r in (False, True)}
    results = {}
    for group, names in models.items():
        for name in names:
            if not (M / f"{name}.pt").exists():
                print("missing", name); continue
            (dl_dev, y_dev), (dl_test, y_test) = loaders["robust" in name]
            model = load_trained_model(M / f"{name}.pt", device, num_classes=5).to(device).eval()
            ordinal = bool(getattr(model, "ordinal", False))
            R = {"group": group}

            def evaluate(mdl, label, shift=None):
                z = logits_of(mdl, dl_test, device)
                if ordinal:
                    b = torch.zeros(z.shape[1]) if shift is None else shift
                    p, probs = decode(z, b), ordinal_to_probs(z + b).numpy()
                else:
                    probs = torch.softmax(z, 1).numpy(); p = probs.argmax(1)
                R[label] = metrics(y_test, p, probs, ci=(label in ("zero_shot", "adabn")))

            evaluate(model, "zero_shot")
            ada = adabn(model, dl_dev, device)
            if ada is not None:
                evaluate(ada, "adabn")
            if ordinal:
                for tag, mdl in (("", model), ("adabn_", ada)):
                    if mdl is None:
                        continue
                    zd = logits_of(mdl, dl_dev, device)
                    for n in (50, 100, len(y_dev)):
                        draws = 5 if n < len(y_dev) else 1
                        vals = []
                        for d in range(draws):
                            idx = np.random.default_rng(d).permutation(len(y_dev))[:n]
                            vals.append(fit_shifts(zd[idx], y_dev[idx]))
                        zt = logits_of(mdl, dl_test, device)
                        ms = []
                        for b in vals:
                            ms.append(metrics(y_test, decode(zt, b), ordinal_to_probs(zt + b).numpy()))
                        avg = {k: float(np.mean([m[k] for m in ms])) for k in ("qwk", "acc", "mae", "within1", "bin_acc", "sens", "spec", "macro_f1", "bin_auc")}
                        avg["recall"] = np.mean([m["recall"] for m in ms], 0).tolist()
                        avg["n_draws"] = draws
                        R[f"{tag}recal_n{n if n < len(y_dev) else 'all'}"] = avg
            results[name] = R
            print(name, {k: round(v["qwk"], 3) for k, v in R.items() if isinstance(v, dict)}, flush=True)
            del model, ada
            torch.cuda.empty_cache()

    out = PROJECT_ROOT / "reports" / ("external_adapt_tierb" if args.tier_b else "external_adapt")
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))

    methods = ["zero_shot", "adabn", "recal_n50", "recal_n100", "recal_nall", "adabn_recal_n100", "adabn_recal_nall"]
    L = ["# External adaptation study (generated by scripts/evaluate_external_adapt.py)", "",
         f"Held-out external TEST part only (n={len(y_test)}); adaptation/recalibration used the DEV part (n={len(y_dev)}). "
         "Mean ± SD over seeds (single run when n=1). 'n/a' = method not applicable (no BatchNorm / non-ordinal).", ""]
    for key, title in (("qwk", "QWK"), ("acc", "Accuracy"), ("bin_acc", "Binary accuracy (KL>=2)"), ("spec", "Specificity (binary)"), ("sens", "Sensitivity (binary)")):
        L += [f"## {title}", "", "| Group | n | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
        for group in models:
            rs = [r for r in results.values() if r["group"] == group]
            if not rs:
                continue
            cells = []
            for m_ in methods:
                v = [r[m_][key] for r in rs if m_ in r]
                cells.append("n/a" if not v else (f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f}"))
            L.append(f"| {group} | {len(rs)} | " + " | ".join(cells) + " |")
        L.append("")
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("wrote reports/external_adapt/results.json and SUMMARY.md")


if __name__ == "__main__":
    main()
