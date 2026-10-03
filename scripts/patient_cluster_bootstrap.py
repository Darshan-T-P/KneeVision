"""Real patient-level cluster bootstrap for test-set metrics.

Reads a per-sample predictions CSV (produced by
`scripts/evaluate_ordinal_test.py --save-predictions`) and computes a
bootstrap confidence interval that resamples PATIENTS (not individual
images) with replacement, so that a patient's two knees don't get counted
as independent evidence.

This is the real version of the computation that
`reports/FINAL_KNEEVISION_RESEARCH_AUDIT.md`'s "Phase 12 — Clustered
Statistical Uncertainty" section describes -- that section's numbers could
NOT have been produced by any code or artifact currently in this repo
(no per-sample predictions file existed anywhere before this script's
companion `--save-predictions` flag was added, there is no bootstrap
script anywhere in the repo, and the notebook contains no bootstrap code),
so they should be treated as unverified until replaced by a real run of
this script.

Usage:
    uv run python scripts/patient_cluster_bootstrap.py \\
        --predictions models/final_ordinal_soft_mixup_a04_test.predictions.csv \\
        --n-boot 1000 --seed 42
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score


def load_predictions(path: Path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "patient_id": r["patient_id"],
                "true": int(r["true_label"]),
                "pred": int(r["pred_label"]),
            })
    return rows


def compute_metrics(trues, preds):
    trues = np.asarray(trues)
    preds = np.asarray(preds)
    errors = np.abs(trues - preds)
    return {
        "qwk": float(cohen_kappa_score(trues, preds, weights="quadratic")),
        "mae": float(errors.mean()),
        "within_1": float((errors <= 1).mean()),
        "exact_accuracy": float((errors == 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = load_predictions(args.predictions)
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(r)
    patients = sorted(by_patient)
    n_patients = len(patients)
    print(f"Loaded {len(rows)} predictions across {n_patients} unique patients from {args.predictions}")

    point = compute_metrics([r["true"] for r in rows], [r["pred"] for r in rows])
    print("Point estimate (whole frozen test set, no resampling):", point)

    rng = np.random.default_rng(args.seed)
    boot_results = defaultdict(list)
    for b in range(args.n_boot):
        sampled_patients = rng.choice(patients, size=n_patients, replace=True)
        trues, preds = [], []
        for p in sampled_patients:
            for r in by_patient[p]:
                trues.append(r["true"])
                preds.append(r["pred"])
        m = compute_metrics(trues, preds)
        for k, v in m.items():
            boot_results[k].append(v)

    summary = {}
    for k, vals in boot_results.items():
        arr = np.asarray(vals)
        summary[k] = {
            "point_estimate": point[k],
            "bootstrap_mean": float(arr.mean()),
            "ci_lower_2.5pct": float(np.percentile(arr, 2.5)),
            "ci_upper_97.5pct": float(np.percentile(arr, 97.5)),
        }

    print(json.dumps(summary, indent=2))

    out_path = args.out or args.predictions.parent.parent / "reports" / f"{args.predictions.stem}_bootstrap.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_patients": n_patients,
        "n_samples": len(rows),
        "source_predictions_file": str(args.predictions),
        "results": summary,
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
