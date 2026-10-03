"""Post-hoc clinical regrouping from an ALREADY-FROZEN 5-class confusion matrix.

This performs ZERO new model inference, ZERO checkpoint loads, and ZERO test
split reloads. It only re-aggregates the 5x5 confusion matrix that a prior,
single, sanctioned test-set evaluation already wrote to
`models/<tag>.json` (field "confusion_matrix"), summing it into coarser
clinical groupings. Because every (true, predicted) pair in the 5-class
matrix is already a complete sufficient statistic for any *deterministic*
relabeling of the 5 grades into fewer groups, this is pure arithmetic on
numbers that were frozen at evaluation time -- it cannot leak, re-optimize
against, or otherwise touch the test set, so it does not violate a
"single frozen test pass" protocol the way re-running inference would.

Uses the SAME grouping definitions as `scripts/evaluate_grouped.py` so a
grouped number computed this way is directly comparable to one computed by
that script for a different checkpoint:
  binary : KL0-1 -> No OA     | KL2-4 -> OA
  3class : KL0-1 -> None/Doubtful | KL2 -> Mild | KL3-4 -> Moderate/Severe

Usage:
    python3 scripts/regroup_from_confusion_matrix.py [--source models/x.json] [--groups binary 3class]
"""
import argparse
import json
from pathlib import Path

GROUPINGS = {
    "binary": {
        "names": ["No OA (KL 0-1)", "OA (KL 2-4)"],
        "members": [(0, 1), (2, 3, 4)],
    },
    "3class": {
        "names": ["None/Doubtful (KL 0-1)", "Mild (KL 2)", "Moderate/Severe (KL 3-4)"],
        "members": [(0, 1), (2,), (3, 4)],
    },
}


def grouped_confusion_matrix(cm5, members):
    """Collapse a 5x5 (true x pred) confusion matrix into len(members) x len(members)
    by summing the rows/columns belonging to each group, in group order."""
    n = len(members)
    grouped = [[0 for _ in range(n)] for _ in range(n)]
    for gi, true_grades in enumerate(members):
        for gj, pred_grades in enumerate(members):
            grouped[gi][gj] = sum(cm5[t][p] for t in true_grades for p in pred_grades)
    return grouped


def metrics_from_confusion_matrix(cm):
    n = len(cm)
    total = sum(sum(row) for row in cm)
    row_sums = [sum(cm[i]) for i in range(n)]
    col_sums = [sum(cm[i][j] for i in range(n)) for j in range(n)]

    per_class = []
    for i in range(n):
        tp = cm[i][i]
        fp = col_sums[i] - tp
        fn = row_sums[i] - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1, "support": row_sums[i]})

    accuracy = sum(cm[i][i] for i in range(n)) / total
    macro_f1 = sum(c["f1"] for c in per_class) / n

    # Quadratic-weighted kappa over the GROUP indices (reduces to standard
    # unweighted Cohen's kappa when n==2, since (i-j)^2/(n-1)^2 is then 0/1).
    po_weighted_disagreement = 0.0
    pe_weighted_disagreement = 0.0
    denom = (n - 1) ** 2 if n > 1 else 1
    for i in range(n):
        for j in range(n):
            w = ((i - j) ** 2) / denom if n > 1 else 0.0
            po_weighted_disagreement += w * cm[i][j] / total
            pe_weighted_disagreement += w * (row_sums[i] * col_sums[j]) / (total * total)
    qwk = 1 - (po_weighted_disagreement / pe_weighted_disagreement) if pe_weighted_disagreement > 0 else float("nan")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "quadratic_weighted_kappa": qwk,
        "per_class": per_class,
        "confusion_matrix": cm,
        "support_total": total,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="models/final_ordinal_soft_mixup_a04_test.json",
                        help="Frozen test-evaluation JSON containing a 5x5 confusion_matrix field")
    parser.add_argument("--groups", nargs="+", default=["binary", "3class"], choices=list(GROUPINGS))
    parser.add_argument("--out", default=None, help="Optional output .md path")
    args = parser.parse_args()

    source_path = Path(args.source)
    data = json.loads(source_path.read_text())
    cm5 = data["confusion_matrix"]
    class_order = data.get("class_order", ["KL0", "KL1", "KL2", "KL3", "KL4"])
    assert len(cm5) == 5 and all(len(row) == 5 for row in cm5), "expected a frozen 5x5 confusion matrix"

    lines = []
    lines.append(f"# Post-hoc clinical regrouping (derived, no new inference)\n")
    lines.append(f"**Source (frozen, untouched):** `{args.source}`")
    lines.append(f"**Source checkpoint SHA-256:** `{data.get('checkpoint_sha256', 'n/a')}`")
    lines.append(f"**Method:** pure arithmetic re-aggregation of the already-frozen {len(cm5)}x{len(cm5)} "
                 f"test confusion matrix ({class_order}). No model inference, no checkpoint reload, "
                 f"no test-split reload was performed to produce this file.\n")

    results = {}
    for group_key in args.groups:
        spec = GROUPINGS[group_key]
        cm_g = grouped_confusion_matrix(cm5, spec["members"])
        m = metrics_from_confusion_matrix(cm_g)
        results[group_key] = {**m, "group_names": spec["names"]}

        lines.append(f"## {group_key} grouping: {spec['names']}\n")
        lines.append(f"- Accuracy: {m['accuracy']*100:.2f}%")
        lines.append(f"- Macro F1: {m['macro_f1']:.4f}")
        lines.append(f"- Quadratic-weighted kappa: {m['quadratic_weighted_kappa']:.4f}")
        lines.append("")
        lines.append("| Group | Precision | Recall | F1 | Support |")
        lines.append("|---|---|---|---|---|")
        for name, c in zip(spec["names"], m["per_class"]):
            lines.append(f"| {name} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} | {c['support']} |")
        lines.append("")
        lines.append(f"Confusion matrix (rows=true, cols=pred, order={spec['names']}):")
        lines.append("```")
        for row in cm_g:
            lines.append(str(row))
        lines.append("```\n")

    report = "\n".join(lines)
    print(report)

    out_path = Path(args.out) if args.out else source_path.parent.parent / "reports" / f"{source_path.stem}_grouped.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    json_out = out_path.with_suffix(".json")
    json_out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path} and {json_out}")


if __name__ == "__main__":
    main()
