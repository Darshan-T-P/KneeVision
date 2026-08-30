"""Download/ingest the OAI clinical dataset (hosted by the NIMH Data Archive, NDA).

The OAI permanent archive is at https://nda.nih.gov/oai

Access is NOT public: you must
  1. create an NDA account  (https://nda.nih.gov/nda/creating-an-nda-account)
  2. log in through the Research Auth Service (eRA Commons / Login.gov / PIV-CAC)
  3. agree to the OAI data-access terms and await approval
Once approved, download the ASCII tables and place them in data/oai/raw/:

    kxr_sq_bu00.txt      baseline X-ray readings (one row per knee, KL grades)
    AllClinical00.txt    baseline clinical variables (age, BMI, WOMAC)

then run:

    uv run python scripts/download_oai.py ingest

The script joins the two tables on subject ID, composes one clinical summary
per knee, splits subjects (never split inside a patient), and writes:

    data/oai/processed/oai_clinical.csv    canonical CSV (report + kl_grade + split)
    data/oai/reports/{split}/{kl_grade}/   folder layout for train_clinical.py

Notes on the real OAI files:
  - kxr_sq_bu00.txt is "|"-delimited with a SIDE column and one row per knee;
    the KL grade is in V00XRKL, encoded like "2: 2" (code : label).
  - AllClinical00.txt is "|"-delimited with ~1,200 columns; this release has
    age (V00AGE), BMI (P01BMI), WOMAC 0-100 items (V00WOMKPR=stiffness,
    V00WOMSTFR=stiffness, V00WOMADLR=limitation). SEX lives in the separate
    Enrollees table, so it is optional.

Usage:
    uv run python scripts/download_oai.py status                  # access steps + file check
    uv run python scripts/download_oai.py ingest [--raw DIR]      # build the dataset
    # override columns if your OAI release differs, e.g.:
    ingest --kl V00XRKL --age V00AGE --bmi P01BMI
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kneevision.config.settings import OAI_DATA_DIR  # noqa: E402
from kneevision.clinical.prepare import compose_clinical_report  # noqa: E402

RAW_DIR = OAI_DATA_DIR / "raw"
PROCESSED_DIR = OAI_DATA_DIR / "processed"
REPORTS_DIR = OAI_DATA_DIR / "reports"
CSV_OUT = PROCESSED_DIR / "oai_clinical.csv"

REQUIRED_FILES = {
    "kxr_sq_bu00.txt": "baseline X-ray readings (KL grades) — from 'Knee X-Ray Image Assessments'",
    "AllClinical00.txt": "baseline clinical variables — from 'AllClinical Dataset'",
}

# Candidate column names per logical field, in order tried; CLI flags override.
# kxr columns are per-knee (SIDE + a single KL). clinical columns carry the
# demographic / symptom features.
DEFAULT_COLS = {
    # -- kxr_sq_bu00.txt --
    "id": ["ID"],
    "side": ["SIDE"],
    "kl": ["V00XRKL", "V00XRLKL", "V00XRKLG"],
    # -- AllClinical00.txt --
    "age": ["V00AGE", "AGE", "P02AGE"],
    "sex": ["P02SEX", "SEX", "V00SEX"],           # optional (Enrollees table)
    "bmi": ["P01BMI", "P02BMI", "BMI", "V00BMI"],
    "pain": ["V00WOMKPR", "V00WOMKPE", "P04WOMKPR", "WOMACPAIN"],
    "stiffness": ["V00WOMSTFR", "V00WOMSTFE", "P04WOMSTFR", "WOMACSTIFF"],
    "function": ["V00WOMADLR", "V00WOMADLE", "P04WOMADLR", "WOMACFUNC"],
    "injury_r": ["P01INJR"],
    "injury_l": ["P01INJL"],
    "surgery_r": ["P01KSURGR"],
    "surgery_l": ["P01KSURGL"],
    "meds": ["P01KPMED"],
}

_ACCESS_STEPS = """\
=======================================================================
  HOW TO GET THE OAI DATASET (free, but gated by NDA approval)
=======================================================================
1. Portal        https://nda.nih.gov/oai
2. Create an NDA account:
     https://nda.nih.gov/nda/creating-an-nda-account
3. Log in via the Research Auth Service (eRA Commons, Login.gov, or PIV/CAC)
4. Agree to the OAI data-access terms (Data Use Certification) and submit.
   Approval typically takes a few days to ~2 weeks.
5. Once approved, download these ASCII tables ('Download Complete Datasets'):
       Knee X-Ray Image Assessments  -> kxr_sq_bu00.txt (KL grade per knee)
       AllClinical Dataset            -> AllClinical00.txt (age, BMI, WOMAC)
   (Optional follow-ups kxr_sq_bu01/03/05/06/08.txt for longitudinal data.)
6. Place the files in:
       {raw}
7. Build the clinical dataset:
       uv run python scripts/download_oai.py ingest
   Then train the clinical-text model:
       uv run python scripts/train_clinical.py --data data/oai/processed/oai_clinical.csv
   or on the folder layout:
       uv run python scripts/train_clinical.py --data data/oai/reports
Notes
  - The OAI archive has no free-text radiology reports. This pipeline builds a
    structured clinical summary (age, BMI, WOMAC pain/stiffness/function) per
    knee and keeps the KL grade as the label — the model predicts severity
    from patient-reported and demographic features.
  - For real narrative reports, see the expert-annotated OAI report dataset on
    IEEE DataPort (DOI 10.21227/vcpg-qm58, subscription required).
  - Cite: https://nda.nih.gov/oai
========================================"""


def _find_col(header, candidates):
    upper = {c.upper(): c for c in header}
    for cand in candidates:
        if cand.upper() in upper:
            return upper[cand.upper()]
    return None


def _read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    """Return (header, rows). Delimiter detected as the most common of , | tab."""
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    sample = "\n".join(lines[:5])
    counts = {d: sample.count(d) for d in (",", "\t", "|")}
    delim = max(counts, key=counts.get) if max(counts.values()) > 0 else None
    header = [c.strip().upper() for c in lines[0].split(delim if delim else None)]
    rows = [line.split(delim) if delim else line.split() for line in lines[1:]]
    return header, rows


def _code(value) -> int | None:
    """Parse an OAI "code: label" field and return the integer code (e.g. \"2: 2\" -> 2)."""
    if value is None:
        return None
    value = str(value).strip()
    if not value or value in {".", "NA", "N/A", "99", "999", "-1"}:
        return None
    try:
        return int(float(value.split(":")[0].strip()))
    except ValueError:
        return None


def _num(value) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value in {".", "NA", "N/A", "99", "999", "-1"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _side_from_code(code) -> str:
    return {"1": "right", "2": "left"}.get(str(code), "right")


def _row_value(row, header, col):
    if col is None or not row or len(row) <= header.index(col):
        return ""
    return row[header.index(col)]


def build_dataset(raw_dir: Path, out_csv: Path, out_reports: Path,
                  cols: dict, split_sizes=(0.7, 0.15, 0.15), seed: int = 42) -> dict:
    """Join kxr_sq_bu00 + AllClinical00 on ID and write the clinical dataset.

    Returns a summary dict with counts for logging/tests.
    """
    kxr_path = raw_dir / "kxr_sq_bu00.txt"
    clin_path = raw_dir / "AllClinical00.txt"
    for p in (kxr_path, clin_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run `status` for access steps.")

    kxr_header, kxr_rows = _read_table(kxr_path)
    clin_header, clin_rows = _read_table(clin_path)

    def resolve(feature, header):
        for cand in cols.get(feature) or []:
            if cand:
                found = _find_col(header, [cand])
                if found:
                    return found
        return None

    kxr_idx = {"id": resolve("id", kxr_header),
               "side": resolve("side", kxr_header),
               "kl": resolve("kl", kxr_header)}
    missing_kxr = [k for k, v in kxr_idx.items() if v is None]
    if missing_kxr:
        raise ValueError(
            f"Missing kxr_sq_bu00 columns {missing_kxr} (tried {DEFAULT_COLS}). "
            f"Header: {kxr_header}\nPass correct names via --id/--side/--kl."
        )

    clin_idx = {f: resolve(f, clin_header) for f in ("age", "sex", "bmi", "pain", "stiffness", "function", "injury_r", "injury_l", "surgery_r", "surgery_l", "meds")}
    if clin_idx["age"] is None:
        raise ValueError(
            f"Could not find an 'age' column in AllClinical00. Header: {clin_header}"
        )
    clin_id_col = resolve("id", clin_header) or (clin_header[0] if clin_header else None)
    if clin_id_col is None:
        raise ValueError("AllClinical00 has no ID column.")

    clinical_by_id = {}
    for row in clin_rows:
        if len(row) < len(clin_header):
            continue
        pid = _row_value(row, clin_header, clin_id_col)
        if not pid:
            continue
        vals = {}
        for feature in clin_idx:
            if clin_idx[feature]:
                vals[feature] = _row_value(row, clin_header, clin_idx[feature])
        clinical_by_id.setdefault(pid, {}).update(vals)

    records = []
    for row in kxr_rows:
        if len(row) < len(kxr_header):
            continue
        pid = _row_value(row, kxr_header, kxr_idx["id"])
        kl = _code(_row_value(row, kxr_header, kxr_idx["kl"]))
        if kl is None or not (0 <= kl <= 4):
            continue
        sc = _code(_row_value(row, kxr_header, kxr_idx["side"])) if kxr_idx.get("side") else None
        side = _side_from_code(sc) if sc is not None else "right"

        clinical = clinical_by_id.get(pid, {})
        features = {
            "age": _num(clinical.get("age")),
            "sex": clinical.get("sex") or None,
            "bmi": _num(clinical.get("bmi")),
            "pain": _num(clinical.get("pain")),
            "stiffness": _num(clinical.get("stiffness")),
            "function": _num(clinical.get("function")),
            "injury": clinical.get("injury_r") if side == "right" else clinical.get("injury_l"),
            "surgery": clinical.get("surgery_r") if side == "right" else clinical.get("surgery_l"),
            "meds": clinical.get("meds"),
        }
        records.append({"id": pid, "side": side, "kl_grade": kl, "features": features})

    ids = sorted({r["id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    a = int(n * split_sizes[0])
    b = a + int(n * split_sizes[1])
    labels = ["train"] * a + ["val"] * (b - a) + ["test"] * (n - b)
    split_of_id = dict(zip(ids, labels))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "side", "split", "kl_grade", "report",
                         "age", "sex", "bmi", "pain", "stiffness", "function",
                         "injury", "surgery", "meds"])
        for r in sorted(records, key=lambda r: (r["id"], r["side"])):
            f_ = r["features"]
            writer.writerow([r["id"], r["side"], split_of_id[r["id"]], r["kl_grade"],
                             compose_clinical_report(f_, side=r["side"]),
                             f_.get("age") or "", f_.get("sex") or "",
                             f_.get("bmi") or "", f_.get("pain") or "",
                             f_.get("stiffness") or "", f_.get("function") or "",
                             f_.get("injury") or "", f_.get("surgery") or "", f_.get("meds") or ""])

    written = 0
    for r in records:
        dest = out_reports / split_of_id[r["id"]] / str(r["kl_grade"])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"report_{r['id']}_{r['side']}.txt") \
            .write_text(compose_clinical_report(r["features"], side=r["side"]))
        written += 1

    counts = {"subjects": len(ids), "knee_records": len(records), "reports_written": written,
              "splits": {s: sum(1 for pid in ids if split_of_id[pid] == s) for s in ("train", "val", "test")}}
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="print access steps and check raw files")

    ingest = sub.add_parser("ingest", help="build the clinical dataset from raw OAI files")
    ingest.add_argument("--raw", type=Path, default=RAW_DIR)
    ingest.add_argument("--out-csv", type=Path, default=CSV_OUT)
    ingest.add_argument("--out-reports", type=Path, default=REPORTS_DIR)
    ingest.add_argument("--seed", type=int, default=42)
    ingest.add_argument("--train", type=float, default=0.7)
    ingest.add_argument("--val", type=float, default=0.15)
    for flag, cand in DEFAULT_COLS.items():
        ingest.add_argument(f"--{flag}", default=cand[0], help=f"column name (default: {cand[0]})")

    args = parser.parse_args()
    if args.command != "ingest":
        print(_ACCESS_STEPS.format(raw=RAW_DIR))
        if args.command != "status":
            print("Unknown command:", args.command)
        return

    cols = {k: [getattr(args, k)] for k in DEFAULT_COLS}
    try:
        counts = build_dataset(args.raw, args.out_csv, args.out_reports, cols,
                               split_sizes=(args.train, args.val, 1 - args.train - args.val),
                               seed=args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(_ACCESS_STEPS.format(raw=RAW_DIR), file=sys.stderr)
        sys.exit(1)

    print(f"Subjects:            {counts['subjects']}")
    print(f"Knee records:        {counts['knee_records']}")
    print(f"Reports written:     {counts['reports_written']}")
    print(f"Subject splits:      {counts['splits']}")
    print(f"CSV:                 {args.out_csv}")
    print(f"Report folders:      {args.out_reports}/{{train,val,test}}/{{0..4}}")
    print("\nNext step:")
    print(f"  uv run python scripts/train_clinical.py --data {args.out_csv} --epochs 10")


if __name__ == "__main__":
    main()