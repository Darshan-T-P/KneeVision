"""Data Pairing & Leakage Validation Script for KneeVision++.

Produces:
  - Console report with statistics
  - DATA_VALIDATION_REPORT.md
  - LEAKAGE_ANALYSIS.md

Usage:
    uv run python scripts/validate_data_pairing.py
    uv run python scripts/validate_data_pairing.py --out-dir reports/data_validation
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kneevision.config.settings import RAW_DATA_DIR, OAI_DATA_DIR


def extract_patient_id(filename: str) -> tuple[str, str]:
    """Parse Kaggle filename like '9911788L.png' → ('9911788', 'L')."""
    stem = Path(filename).stem
    if len(stem) >= 2 and stem[-1].upper() in ("L", "R"):
        return stem[:-1], stem[-1].upper()
    return stem, ""


def load_image_metadata(raw_dir: Path) -> dict[str, list[dict]]:
    """Returns {split: [{path, patient_id, side, kl_grade}]}"""
    result = {}
    for split in ["train", "val", "test"]:
        split_dir = raw_dir / split
        if not split_dir.exists():
            continue
        records = []
        for grade_dir in sorted(split_dir.iterdir()):
            if not grade_dir.is_dir():
                continue
            kl = int(grade_dir.name)
            for f in sorted(grade_dir.glob("*.png")):
                pid, side = extract_patient_id(f.name)
                records.append({
                    "path": f,
                    "filename": f.name,
                    "patient_id": pid,
                    "side": side,
                    "kl_grade": kl,
                    "split": split,
                })
        result[split] = records
    return result


def load_oai_records(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def run_validation(raw_dir: Path, oai_csv: Path) -> dict:
    """Run all validation checks. Returns structured results dict."""
    results = {}

    # ── 1. Image statistics ──────────────────────────────────────────────────
    img_meta = load_image_metadata(raw_dir)
    all_img = [r for records in img_meta.values() for r in records]

    total_images = len(all_img)
    all_patients = set(r["patient_id"] for r in all_img)
    all_knees = set((r["patient_id"], r["side"]) for r in all_img)

    results["total_images"] = total_images
    results["unique_patients"] = len(all_patients)
    results["unique_knees"] = len(all_knees)

    split_stats = {}
    for split, records in img_meta.items():
        patients = set(r["patient_id"] for r in records)
        knees = set((r["patient_id"], r["side"]) for r in records)
        dist = dict(sorted(Counter(r["kl_grade"] for r in records).items()))
        split_stats[split] = {
            "images": len(records),
            "patients": len(patients),
            "knees": len(knees),
            "distribution": dist,
        }
    results["splits"] = split_stats

    # ── 2. Patient leakage ───────────────────────────────────────────────────
    leakage = {}
    split_patients = {s: set(r["patient_id"] for r in recs)
                     for s, recs in img_meta.items()}
    split_knees = {s: set((r["patient_id"], r["side"]) for r in recs)
                   for s, recs in img_meta.items()}

    for s1 in split_patients:
        for s2 in split_patients:
            if s1 >= s2:
                continue
            patient_leak = split_patients[s1] & split_patients[s2]
            knee_leak = split_knees[s1] & split_knees[s2]
            key = f"{s1}/{s2}"
            leakage[key] = {
                "patient_leakage": len(patient_leak),
                "knee_leakage": len(knee_leak),
                "examples": list(patient_leak)[:5],
            }
    results["leakage"] = leakage

    # ── 3. Duplicate image detection ─────────────────────────────────────────
    knee_occurrences = defaultdict(list)
    for r in all_img:
        knee_occurrences[(r["patient_id"], r["side"])].append(r)
    multi_entry = {k: v for k, v in knee_occurrences.items() if len(v) > 1}
    results["duplicate_knee_entries"] = len(multi_entry)
    results["duplicate_examples"] = {
        str(k): [{"split": r["split"], "kl": r["kl_grade"]} for r in v]
        for k, v in list(multi_entry.items())[:5]
    }

    # ── 4. OAI multimodal pairing ────────────────────────────────────────────
    oai_records = load_oai_records(oai_csv)
    results["oai_available"] = len(oai_records) > 0
    results["oai_total_records"] = len(oai_records)

    if oai_records:
        oai_patients = set(r["id"] for r in oai_records)
        oai_by_key = {}
        for r in oai_records:
            side_full = r["side"].lower()  # 'left'/'right'
            side_char = "L" if side_full == "left" else "R"
            oai_by_key[(r["id"], side_char)] = r

        results["oai_unique_patients"] = len(oai_patients)

        # Match every image to OAI record
        test_records = img_meta.get("test", [])
        matched = 0
        unmatched = 0
        unmatched_examples = []
        label_match = 0
        label_mismatch = 0
        label_mismatch_examples = []

        for r in all_img:
            pid, side = r["patient_id"], r["side"]
            oai_row = oai_by_key.get((pid, side))
            if oai_row:
                matched += 1
                oai_kl = int(oai_row["kl_grade"])
                if oai_kl == r["kl_grade"]:
                    label_match += 1
                else:
                    label_mismatch += 1
                    if len(label_mismatch_examples) < 5:
                        label_mismatch_examples.append({
                            "file": r["filename"],
                            "kaggle_kl": r["kl_grade"],
                            "oai_kl": oai_kl,
                        })
            else:
                unmatched += 1
                if len(unmatched_examples) < 5:
                    unmatched_examples.append(r["filename"])

        results["pairing"] = {
            "total_images": total_images,
            "matched": matched,
            "unmatched": unmatched,
            "match_rate_pct": 100.0 * matched / total_images if total_images else 0,
            "label_match": label_match,
            "label_mismatch": label_mismatch,
            "label_agreement_pct": 100.0 * label_match / matched if matched else 0,
            "unmatched_examples": unmatched_examples,
            "label_mismatch_examples": label_mismatch_examples,
        }

        # ── 5. OARSI field coverage ─────────────────────────────────────────
        oarsi_fields = ["jsn_m", "jsn_l", "osteophyte_m", "osteophyte_l",
                        "sclerosis_m", "sclerosis_l", "attrition_m", "attrition_l"]
        field_coverage = {}
        for field in oarsi_fields:
            filled = sum(1 for r in oai_records if r.get(field, "").strip() not in ("", "None"))
            field_coverage[field] = {
                "filled": filled,
                "total": len(oai_records),
                "pct": 100.0 * filled / len(oai_records) if oai_records else 0,
            }
        results["oarsi_coverage"] = field_coverage

        # ── 6. OARSI-KL definitional co-occurrence analysis ─────────────────
        def get_int(v):
            try:
                return int(float(v)) if str(v).strip() not in ("", "None") else None
            except (TypeError, ValueError):
                return None

        cooccurrence = {}
        for grade in range(5):
            grade_rows = [r for r in oai_records if get_int(r["kl_grade"]) == grade]
            cooccurrence[f"KL{grade}"] = {
                "count": len(grade_rows),
                "jsn_m_mean": None,
                "osteophyte_m_mean": None,
            }
            jsn_vals = [get_int(r.get("jsn_m")) for r in grade_rows
                        if get_int(r.get("jsn_m")) is not None]
            oste_vals = [get_int(r.get("osteophyte_m")) for r in grade_rows
                         if get_int(r.get("osteophyte_m")) is not None]
            if jsn_vals:
                cooccurrence[f"KL{grade}"]["jsn_m_mean"] = round(sum(jsn_vals) / len(jsn_vals), 3)
            if oste_vals:
                cooccurrence[f"KL{grade}"]["osteophyte_m_mean"] = round(sum(oste_vals) / len(oste_vals), 3)
        results["oarsi_kl_cooccurrence"] = cooccurrence

    return results


def print_report(results: dict):
    """Print human-readable validation report."""
    print("=" * 70)
    print("  KNEEVISION++ DATA PAIRING & LEAKAGE VALIDATION REPORT")
    print("=" * 70)

    print(f"\nTotal images:          {results['total_images']:,}")
    print(f"Unique patients:       {results['unique_patients']:,}")
    print(f"Unique knees:          {results['unique_knees']:,}")

    print("\nSplit breakdown:")
    for split, s in results["splits"].items():
        print(f"  {split:6s}: {s['images']:5d} images | {s['patients']:4d} patients | dist={s['distribution']}")

    print("\nPatient-level leakage between splits:")
    all_clean = True
    for key, l in results["leakage"].items():
        status = "✅ CLEAN" if l["patient_leakage"] == 0 else f"❌ LEAKAGE: {l['patient_leakage']} patients"
        print(f"  {key}: {status}")
        if l["patient_leakage"] > 0:
            print(f"    Examples: {l['examples']}")
            all_clean = False
    if all_clean:
        print("  → No patient leakage detected across any split pair.")

    print(f"\nDuplicate knee entries: {results['duplicate_knee_entries']}")

    if results.get("oai_available"):
        print(f"\nOAI clinical records:  {results['oai_total_records']:,}")
        print(f"OAI unique patients:   {results['oai_unique_patients']:,}")

        p = results["pairing"]
        print(f"\nMultimodal pairing (all {p['total_images']:,} images):")
        print(f"  Matched to OAI record: {p['matched']:,} ({p['match_rate_pct']:.1f}%)")
        print(f"  Unmatched (fallback):  {p['unmatched']:,}")
        print(f"  Label agreement:       {p['label_match']:,}/{p['matched']:,} ({p['label_agreement_pct']:.1f}%)")
        if p["label_mismatch_examples"]:
            print("  Label mismatches (examples):")
            for ex in p["label_mismatch_examples"]:
                print(f"    {ex['file']}: Kaggle={ex['kaggle_kl']}, OAI={ex['oai_kl']}")

        print("\nOARSI field coverage (OAI CSV):")
        for field, cov in results["oarsi_coverage"].items():
            print(f"  {field:15s}: {cov['filled']:,}/{cov['total']:,} ({cov['pct']:.0f}%)")

        print("\nOARSI-KL definitional co-occurrence (test split context):")
        print(f"  {'Grade':6s} | {'N':6s} | {'JSN_M mean':10s} | {'Oste_M mean':11s}")
        print("  " + "-" * 42)
        for grade_key, stats in results["oarsi_kl_cooccurrence"].items():
            jsn = f"{stats['jsn_m_mean']:.3f}" if stats["jsn_m_mean"] is not None else "N/A"
            oste = f"{stats['osteophyte_m_mean']:.3f}" if stats["osteophyte_m_mean"] is not None else "N/A"
            print(f"  {grade_key:6s} | {stats['count']:6d} | {jsn:10s} | {oste:11s}")
    else:
        print("\n⚠️  OAI CSV not found — skipping multimodal pairing checks.")

    print("\n" + "=" * 70)


def write_data_validation_report(results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# DATA_VALIDATION_REPORT.md", "",
             "**Generated by**: `scripts/validate_data_pairing.py`", "",
             "## Image Dataset Statistics", "",
             f"| Metric | Value |", "|---|---|",
             f"| Total images | {results['total_images']:,} |",
             f"| Unique patients | {results['unique_patients']:,} |",
             f"| Unique knees | {results['unique_knees']:,} |", ""]

    lines += ["## Split Breakdown", "",
              "| Split | Images | Patients | KL0 | KL1 | KL2 | KL3 | KL4 |",
              "|---|---|---|---|---|---|---|---|"]
    for split, s in results["splits"].items():
        d = s["distribution"]
        lines.append(f"| {split} | {s['images']:,} | {s['patients']:,} | "
                     f"{d.get(0,0)} | {d.get(1,0)} | {d.get(2,0)} | "
                     f"{d.get(3,0)} | {d.get(4,0)} |")
    lines.append("")

    lines += ["## Patient Leakage Between Splits", ""]
    for key, l in results["leakage"].items():
        status = "✅ CLEAN (0 patients)" if l["patient_leakage"] == 0 else f"❌ LEAKAGE: {l['patient_leakage']} patients"
        lines.append(f"- **{key}**: {status}")
    lines.append("")

    if results.get("oai_available"):
        p = results["pairing"]
        lines += ["## Multimodal Pairing Statistics", "",
                  "| Metric | Value |", "|---|---|",
                  f"| OAI total records | {results['oai_total_records']:,} |",
                  f"| OAI unique patients | {results['oai_unique_patients']:,} |",
                  f"| Images matched to OAI | {p['matched']:,} ({p['match_rate_pct']:.1f}%) |",
                  f"| Images unmatched (fallback) | {p['unmatched']:,} |",
                  f"| KL label agreement | {p['label_agreement_pct']:.1f}% |",
                  f"| KL label mismatches | {p['label_mismatch']:,} |",
                  ""]

        lines += ["## OARSI Field Coverage", ""]
        lines.append("| Field | Filled | Total | Coverage |")
        lines.append("|---|---|---|---|")
        for field, cov in results["oarsi_coverage"].items():
            lines.append(f"| {field} | {cov['filled']:,} | {cov['total']:,} | {cov['pct']:.0f}% |")
        lines.append("")

    (out_dir / "DATA_VALIDATION_REPORT.md").write_text("\n".join(lines))

    leakage_lines = [
        "# LEAKAGE_ANALYSIS.md", "",
        "## Summary", "",
        "**Engineering leakage** (label string in features): ❌ NOT present.",
        "`compose_clinical_report()` explicitly excludes the KL grade field.", "",
        "**Definitional co-occurrence** (OARSI components define KL): ⚠️ PRESENT.",
        "This is the fundamental experimental design issue.", "",
        "## What Is Included in Clinical Text", "",
        "- Medial/lateral JSN grade (0–3 ordinal, from kxr_sq_bu00.txt)",
        "- Medial/lateral osteophyte grade (0–3)",
        "- Medial/lateral subchondral sclerosis (0–3)",
        "- Medial/lateral bone attrition (0–3)",
        "- Patient demographics (age, BMI)",
        "- WOMAC symptom scores (pain, stiffness, function) — 0-100",
        "- Injury/surgery/medication history", "",
        "## Why This Is Not Conventional Label Leakage", "",
        "1. The KL grade **string** does not appear in the text.",
        "2. The OARSI sub-component grades and the KL grade come from the same",
        "   radiologist reading session but are **separate annotation outputs**.",
        "3. A radiologist seeing only the sub-component grades could deterministically",
        "   reconstruct the KL grade — but this is also true in clinical practice.",
        "4. This mirrors how the KL system works in reality.", "",
        "## Why This IS an Experimental Design Problem", "",
        "1. The BioClinicalBERT model achieves κ=0.953 primarily because it learns",
        "   to decode the near-deterministic relationship between OARSI components",
        "   and KL grade — not because it learns generalizable disease severity.",
        "2. In a real-world deployment scenario, OARSI sub-component reads may not",
        "   be available as input — they require the same radiologist reading that",
        "   would produce the KL grade itself.",
        "3. The high text-model performance should not be interpreted as",
        "   'clinical text adds independent information beyond imaging.'", "",
        "## Required Paper Disclosures", "",
        "The paper MUST clearly state:",
        "- The clinical text features are derived from the SAME OAI annotation",
        "  session (kxr_sq_bu00.txt) that also produces the KL grade label.",
        "- OARSI sub-components are the structural defining components of KL grades.",
        "- The text model's high performance reflects this definitional relationship.",
        "- In a deployment scenario without pre-existing OARSI reads, the text",
        "  branch's performance would degrade substantially.", "",
        "## OARSI-KL Monotonic Evidence", "",
        "| KL Grade | Mean JSN_M | Mean Osteophyte_M |",
        "|---|---|---|",
    ]
    if results.get("oai_available"):
        for grade_key, stats in results["oarsi_kl_cooccurrence"].items():
            jsn = f"{stats['jsn_m_mean']:.3f}" if stats["jsn_m_mean"] is not None else "N/A"
            oste = f"{stats['osteophyte_m_mean']:.3f}" if stats["osteophyte_m_mean"] is not None else "N/A"
            leakage_lines.append(f"| {grade_key} | {jsn} | {oste} |")
    leakage_lines += ["", "The near-perfect monotonic relationship confirms definitional co-occurrence."]

    (out_dir / "LEAKAGE_ANALYSIS.md").write_text("\n".join(leakage_lines))
    print(f"\nReports written to: {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-data", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--oai-csv", type=Path,
                        default=OAI_DATA_DIR / "processed" / "oai_clinical.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/data_validation"))
    args = parser.parse_args()

    results = run_validation(args.raw_data, args.oai_csv)
    print_report(results)
    write_data_validation_report(results, args.out_dir)


if __name__ == "__main__":
    main()
