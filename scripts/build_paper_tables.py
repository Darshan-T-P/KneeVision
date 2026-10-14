"""Build the complete paper metric tables from saved per-sample test predictions.

Input : reports/test_evals/*.predictions.csv (+ models/final_ordinal_soft_mixup_a04_test.predictions.csv)
Output: reports/PAPER_TABLES.md, reports/paper_metrics.csv, reports/paper_metrics.json

For every checkpoint: QWK / linear kappa / MAE / within-1,2 / accuracy / macro P-R-F1,
per-class P-R-F1 and one-vs-rest AUC, ECE, Brier, binary (KL0-1 vs KL2-4: accuracy,
sensitivity, specificity, AUC) and 3-class (KL0-1 / KL2 / KL3-4) accuracy + QWK,
parameter count, best validation epoch/QWK, and PATIENT-CLUSTER bootstrap 95% CIs
(resample patients, B=1000) for the headline metrics. Nothing is hard-coded: every value is
recomputed from predictions and cross-checked against the stored evaluation JSON.
Uses only test predictions already produced by the single frozen test pass of each model.

Usage: uv run python scripts/build_paper_tables.py [--n-boot 1000]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kneevision.evaluation.report import compute_calibration_metrics  # repo-defined class-wise ECE

ROOT = Path(__file__).resolve().parents[1]
TE = ROOT / "reports" / "test_evals"
MD = ROOT / "models"
K = 5


def load(path):
    pid, y, p, prob = [], [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            pid.append(r["patient_id"]); y.append(int(r["true_label"])); p.append(int(r["pred_label"]))
            prob.append([float(r[f"prob_KL{k}"]) for k in range(K)])
    return np.array(pid), np.array(y), np.array(p), np.array(prob)


def cm_of(y, p, k=K):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    return cm


def qwk_from_cm(cm, weights="quadratic"):
    k = cm.shape[0]
    i, j = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    w = (i - j) ** 2 if weights == "quadratic" else np.abs(i - j)
    n = cm.sum()
    exp = np.outer(cm.sum(1), cm.sum(0)) / n
    den = (w * exp).sum()
    return 1.0 - (w * cm).sum() / den if den > 0 else float("nan")


def prf(cm):
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return prec, rec, f1


def headline(y, p):
    cm = cm_of(y, p)
    prec, rec, f1 = prf(cm)
    d = np.abs(y - p)
    return {
        "qwk": qwk_from_cm(cm), "mae": d.mean(), "acc": np.trace(cm) / cm.sum(),
        "macro_f1": f1.mean(), "within1": (d <= 1).mean(), "kl3_recall": rec[3], "kl1_recall": rec[1],
    }


def ece(prob, y, bins=10):
    conf, pred = prob.max(1), prob.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return tot


def group3(a):
    return np.where(a <= 1, 0, np.where(a == 2, 1, 2))


def full_metrics(y, p, prob):
    cm = cm_of(y, p)
    prec, rec, f1 = prf(cm)
    d = np.abs(y - p)
    onehot = np.eye(K)[y]
    m = {
        "n": len(y), "qwk": qwk_from_cm(cm), "linear_kappa": qwk_from_cm(cm, "linear"), "mae": d.mean(),
        "within1": (d <= 1).mean(), "within2": (d <= 2).mean(), "acc": np.trace(cm) / cm.sum(),
        "macro_precision": prec.mean(), "macro_recall": rec.mean(), "macro_f1": f1.mean(),
        "ece_top1": ece(prob, y), "ece_classwise": compute_calibration_metrics(y, prob)["ece"], "brier_multiclass": ((prob - onehot) ** 2).sum(1).mean(),
        "extreme_errors_ge3": int((d >= 3).sum()), "under_grade_pct_of_errors": float((p < y).sum() / max((p != y).sum(), 1)),
    }
    for k in range(K):
        m[f"KL{k}_precision"], m[f"KL{k}_recall"], m[f"KL{k}_f1"] = prec[k], rec[k], f1[k]
        m[f"KL{k}_auc"] = roc_auc_score((y == k).astype(int), prob[:, k])
    m["macro_auc"] = float(np.mean([m[f"KL{k}_auc"] for k in range(K)]))
    yb, pb = (y >= 2).astype(int), (p >= 2).astype(int)
    tp, tn = ((yb == 1) & (pb == 1)).sum(), ((yb == 0) & (pb == 0)).sum()
    m["bin_acc"] = (tp + tn) / len(y)
    m["bin_sensitivity"], m["bin_specificity"] = tp / max((yb == 1).sum(), 1), tn / max((yb == 0).sum(), 1)
    m["bin_auc"] = roc_auc_score(yb, prob[:, 2:].sum(1))
    y3, p3 = group3(y), group3(p)
    cm3 = cm_of(y3, p3, 3)
    m["tri_acc"], m["tri_qwk"] = np.trace(cm3) / cm3.sum(), qwk_from_cm(cm3)
    return m


def cluster_ci(pid, y, p, n_boot, seed=42):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, g in enumerate(pid):
        groups[g].append(i)
    idx = [np.array(v) for v in groups.values()]
    keys = ["qwk", "mae", "acc", "macro_f1", "within1", "kl3_recall", "kl1_recall"]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx))
        sel = np.concatenate([idx[i] for i in pick])
        h = headline(y[sel], p[sel])
        for k in keys:
            samples[k].append(h[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in samples.items()}


def label_for(stem):
    s = stem.replace("best_", "").replace(".predictions", "")
    if s == "final_ordinal_soft_mixup_a04_test":
        return "DenseNet121 · soft CORAL + MixUp 0.4 · seed 42 (locked)"
    s = s.replace("densenet121_ordinal_short40_base_seed", "DenseNet121 · soft CORAL + MixUp 0.4 · 40-epoch schedule (matched baseline) · seed ")
    s = s.replace("densenet121_ordinal_short40_robust_seed", "DenseNet121 · soft CORAL + MixUp 0.4 · 40-epoch schedule · DOMAIN-ROBUST aug · seed ")
    s = s.replace("densenet121_ordinal_ordinal_soft_mixup_a04_pre_hist_eq", "DenseNet121 · soft CORAL + MixUp 0.4 · histogram-equalization preprocessing · seed 42")
    s = s.replace("densenet121_ordinal_ordinal_soft_mixup_a04_seed", "DenseNet121 · soft CORAL + MixUp 0.4 · seed ")
    s = s.replace("densenet121_multitask_ordinal_soft_mixup_a04_seed", "DenseNet121 + OARSI aux heads · soft CORAL + MixUp 0.4 · seed ")
    s = s.replace("densenet121_multitask_ordinal_soft_mixup_a04", "DenseNet121 + OARSI aux heads · soft CORAL + MixUp 0.4 · seed 42")
    s = s.replace("densenet121_focal_nomixup_seed", "DenseNet121 · Focal, no MixUp · seed ")
    s = s.replace("densenet121_focal_nomixup", "DenseNet121 · Focal, no MixUp · seed 42")
    s = s.replace("densenet121_focal_mixup", "DenseNet121 · Focal + MixUp 0.2 · seed 42")
    s = s.replace("densenet121_ordinal_ordinal_soft_nomixup", "DenseNet121 · soft CORAL, no MixUp · seed 42")
    s = s.replace("densenet121_ordinal_ordinal_soft_mixup", "DenseNet121 · soft CORAL + MixUp 0.2 · seed 42")
    s = s.replace("densenet121", "DenseNet121 · standard 5-class Focal (baseline)")
    for a, b in (("convnext_tiny", "ConvNeXt-Tiny"), ("efficientnet-b4", "EfficientNet-B4"), ("swin_t", "Swin-T"), ("vit_b_16", "ViT-B/16")):
        if s.startswith(a):
            return f"{b} · soft CORAL + MixUp 0.4 · seed 42"
    return s


def ckpt_stem(pred_stem):
    s = pred_stem.replace(".predictions", "")
    return "final_ordinal_soft_mixup_a04" if s == "final_ordinal_soft_mixup_a04_test" else s


def params_and_meta(stem):
    out = {"params_m": None, "best_epoch": None, "val_qwk": None}
    js = MD / f"{stem}.json"
    if js.exists():
        d = json.loads(js.read_text())
        out["best_epoch"], out["val_qwk"] = d.get("epoch"), d.get("best_validation_qwk", d.get("best_kappa"))
    pt = MD / f"{stem}.pt"
    if pt.exists():
        import torch
        sd = torch.load(pt, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        out["params_m"] = sum(v.numel() for v in sd.values() if hasattr(v, "numel")) / 1e6
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    files = sorted(TE.glob("*.predictions.csv")) + [MD / "final_ordinal_soft_mixup_a04_test.predictions.csv"]
    rows = []
    for f in files:
        if not f.exists():
            continue
        pid, y, p, prob = load(f)
        stem = f.name[: -len(".csv")] if f.name.endswith(".csv") else f.stem
        base = f.name.replace(".csv", "")
        m = full_metrics(y, p, prob)
        # cross-check against the stored evaluation JSON from the frozen test pass
        cs = ckpt_stem(base)
        js = (TE / f"{cs}.test.json") if (TE / f"{cs}.test.json").exists() else (MD / f"{cs}_test.json")
        if js.exists():
            ref = json.loads(js.read_text())
            assert abs(ref["qwk"] - m["qwk"]) < 1e-6 and abs(ref["mae"] - m["mae"]) < 1e-6 and abs(ref["ece"] - m["ece_classwise"]) < 1e-6, f"mismatch for {cs}"
        m["ci"] = cluster_ci(pid, y, p, args.n_boot)
        m.update(params_and_meta("best_densenet121_ordinal_ordinal_soft_mixup_a04" if cs == "final_ordinal_soft_mixup_a04" else cs))
        m["model"], m["checkpoint"] = label_for(base), cs
        rows.append(m)
        print("ok", m["model"], round(m["qwk"], 4))
    rows.sort(key=lambda r: -r["qwk"])

    def ci(r, k, pct=False):
        lo, hi = r["ci"][k]
        v = r[k if k in r else {"kl3_recall": "KL3_recall", "kl1_recall": "KL1_recall", "within1": "within1"}[k]]
        s = 100 if pct else 1
        f = "{:.1f}" if pct else "{:.3f}"
        return f"{f.format(v * s)} [{f.format(lo * s)}, {f.format(hi * s)}]"

    L = ["# Paper tables (generated by scripts/build_paper_tables.py)", "",
         "All values are recomputed from per-sample predictions of the single frozen test pass (N=1,656; 828 patients) and "
         f"cross-checked against each stored evaluation JSON. 95% CIs: patient-cluster bootstrap, B={args.n_boot}. " +
         "Sorted by test QWK. Single seeds unless the seed table (`SEED_ROBUSTNESS.md`) is cited; differences smaller than the "
         "seed-to-seed SD (~0.013 QWK) should not be called significant.", "",
         "## Table A — Headline ordinal metrics", "",
         "| Model | Params (M) | Val QWK | Test QWK [95% CI] | MAE [95% CI] | Within-1 % [95% CI] | Acc % [95% CI] | Macro-F1 [95% CI] | Linear κ | Within-2 % | Extreme (≥3) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['params_m']:.1f} | {r['val_qwk']:.4f} | {ci(r,'qwk')} | {ci(r,'mae')} | {ci(r,'within1',True)} | "
                 f"{ci(r,'acc',True)} | {ci(r,'macro_f1')} | {r['linear_kappa']:.3f} | {r['within2']*100:.2f} | {r['extreme_errors_ge3']} |"
                 if r["params_m"] is not None and r["val_qwk"] is not None else
                 f"| {r['model']} | n/a | n/a | {ci(r,'qwk')} | {ci(r,'mae')} | {ci(r,'within1',True)} | {ci(r,'acc',True)} | {ci(r,'macro_f1')} | {r['linear_kappa']:.3f} | {r['within2']*100:.2f} | {r['extreme_errors_ge3']} |")
    L += ["", "## Table B — Per-class recall / precision / F1 / one-vs-rest AUC", "",
          "| Model | " + " | ".join(f"KL{k} R / P / F1 / AUC" for k in range(K)) + " | Macro-AUC |", "|---|" + "---|" * 6]
    for r in rows:
        cells = [f"{r[f'KL{k}_recall']:.2f} / {r[f'KL{k}_precision']:.2f} / {r[f'KL{k}_f1']:.2f} / {r[f'KL{k}_auc']:.3f}" for k in range(K)]
        L.append(f"| {r['model']} | " + " | ".join(cells) + f" | {r['macro_auc']:.3f} |")
    L += ["", "## Table C — Clinically grouped tasks and calibration", "",
          "| Model | Binary OA (KL≥2): Acc / Sens / Spec / AUC | 3-class (0-1 / 2 / 3-4): Acc / QWK | KL1 recall [CI] | KL3 recall [CI] | ECE top-label | ECE class-wise (repo) | Brier (multiclass) | Under-grade % of errors |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['bin_acc']*100:.1f} / {r['bin_sensitivity']*100:.1f} / {r['bin_specificity']*100:.1f} / {r['bin_auc']:.3f} | "
                 f"{r['tri_acc']*100:.1f} / {r['tri_qwk']:.3f} | {ci(r,'kl1_recall',True)} | {ci(r,'kl3_recall',True)} | {r['ece_top1']:.3f} | {r['ece_classwise']:.4f} | "
                 f"{r['brier_multiclass']:.3f} | {r['under_grade_pct_of_errors']*100:.0f} |")
    (ROOT / "reports" / "PAPER_TABLES.md").write_text("\n".join(L) + "\n")
    flat = [{k: v for k, v in r.items() if k != "ci"} | {f"{k}_ci_lo": r["ci"][k][0] for k in r["ci"]} | {f"{k}_ci_hi": r["ci"][k][1] for k in r["ci"]} for r in rows]
    with open(ROOT / "reports" / "paper_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        w.writeheader(); w.writerows(flat)
    (ROOT / "reports" / "paper_metrics.json").write_text(json.dumps(flat, indent=2, default=float))
    print(f"wrote reports/PAPER_TABLES.md, paper_metrics.csv/json ({len(rows)} models)")


if __name__ == "__main__":
    main()
