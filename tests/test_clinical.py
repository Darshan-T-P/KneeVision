
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from kneevision.clinical.prepare import (
    generate_report,
    generate_synthetic_dataset,
    load_reports_from_grades,
    load_reports_from_folders,
    load_reports_csv,
    build_synthetic_splits,
    compose_clinical_report,
    compose_radiographic_findings,
)
from kneevision.clinical.model import _infer_ordinal as _infer_clinical_ordinal
from download_oai import build_dataset


def test_generate_report_contains_grade():
    for grade in range(5):
        report = generate_report(grade)
        assert f"KL grade {grade}" in report
        assert "FINDINGS" in report and "IMPRESSION" in report


def test_generate_report_reproducible():
    assert generate_report(3, seed=7) == generate_report(3, seed=7)


def test_generate_and_load_synthetic(tmp_path):
    labels = [0, 0, 2, 4]
    generate_synthetic_dataset(labels, tmp_path)
    texts, loaded = load_reports_from_grades(tmp_path)
    assert sorted(loaded) == labels
    assert len(texts) == len(labels)


def test_load_reports_from_folders(tmp_path):
    d = tmp_path / "train" / "1"
    d.mkdir(parents=True)
    (d / "a.txt").write_text("report text")
    splits = load_reports_from_folders(tmp_path)
    assert splits["train"] == (["report text"], [1])


def test_load_reports_csv(tmp_path):
    p = tmp_path / "reports.csv"
    p.write_text("report,kl_grade,split\nSevere OA,4,train\nNormal,0,test\n")
    splits = load_reports_csv(p)
    assert splits["train"] == (["Severe OA"], [4])
    assert splits["test"] == (["Normal"], [0])


def test_build_synthetic_splits(tmp_path):
    image_splits = {"train": ([0, 1], [0, 2]), "val": ([2], [4])}
    splits = build_synthetic_splits(image_splits, tmp_path)
    assert sorted(splits["train"][1]) == [0, 2]
    assert sorted(splits["val"][1]) == [4]


def test_compose_clinical_report():
    text = compose_clinical_report({"age": 62, "sex": "female", "bmi": 28.4, "pain": 33.5})
    assert "62 years old" in text and "female" in text and "BMI 28.4" in text
    assert "WOMAC pain score is 33.5" in text and "mild to moderate" in text
    assert "KL" not in text  # no label leakage


def test_compose_clinical_report_missing_fields():
    text = compose_clinical_report({}, side="left")
    assert "left knee" in text and "Patient is" not in text


def test_compose_radiographic_findings_real_grades():
    text = compose_radiographic_findings({
        "jsn_m": 2, "osteophyte_m": 3, "sclerosis_m": 1,
        "jsn_l": 0, "osteophyte_l": 0, "sclerosis_l": 0,
        "attrition_m": 2,
    })
    assert "RADIOGRAPHIC FINDINGS" in text
    assert "Medial compartment: moderate joint space narrowing, large osteophytes, mild subchondral sclerosis." in text
    assert "Lateral compartment: no joint space narrowing, no osteophytes, no subchondral sclerosis." in text
    assert "attritional bone change is noted medially" in text
    assert "KL" not in text  # no label leakage


def test_compose_radiographic_findings_handles_missing_and_string_values():
    # Empty-string cells (as read from a CSV DictReader) must be treated as missing.
    assert compose_radiographic_findings({}) == ""
    assert compose_radiographic_findings({"jsn_m": "", "osteophyte_m": "", "sclerosis_m": ""}) == ""
    text = compose_radiographic_findings({"jsn_m": "1", "attrition_l": "3"})
    assert "mild joint space narrowing" in text
    assert "Severe attritional (bone-on-bone) change is noted laterally" in text


def test_compose_clinical_report_includes_radiographic_findings():
    text = compose_clinical_report({"age": 62, "jsn_m": 3, "osteophyte_m": 2, "sclerosis_m": 1})
    assert "RADIOGRAPHIC FINDINGS" in text
    assert "severe joint space narrowing" in text
    assert "KL" not in text


def test_infer_clinical_ordinal_from_checkpoint_shape():
    # Regression test: fusion/eval/streamlit code previously assumed a fixed
    # ordinal setting instead of reading it off the checkpoint, which crashed
    # fusion training whenever the clinical model was retrained non-ordinal.
    ordinal_state = {"classifier.weight": torch.randn(4, 768), "classifier.bias": torch.randn(4)}
    assert _infer_clinical_ordinal(ordinal_state, num_classes=5) is True

    non_ordinal_state = {"classifier.weight": torch.randn(5, 768), "classifier.bias": torch.randn(5)}
    assert _infer_clinical_ordinal(non_ordinal_state, num_classes=5) is False

    assert _infer_clinical_ordinal({}, num_classes=5) is False


def _make_oai_raw(tmp_path, n=30):
    raw = tmp_path / "raw"
    raw.mkdir()
    kxr = ["ID|SIDE|V00XRKL"]
    clin = ["ID|V00AGE|P01BMI|V00WOMKPR|V00WOMSTFR|V00WOMADLR"]
    for i in range(n):
        kxr.append(f"{i:05d}|1: Right|{i % 5}: {i % 5}")
        kxr.append(f"{i:05d}|2: Left|{4 - (i % 5)}: {4 - (i % 5)}")
        clin.append(f"{i:05d}|{55 + i % 20}|{22 + i % 10}.0|{i * 3 % 100}.0|{i * 2 % 100}.0|{i % 100}.0")
    (raw / "kxr_sq_bu00.txt").write_text("\n".join(kxr) + "\n")
    (raw / "AllClinical00.txt").write_text("\n".join(clin) + "\n")
    return raw


def test_build_dataset(tmp_path):
    raw = _make_oai_raw(tmp_path)
    cols = {"id": ["ID"], "side": ["SIDE"], "kl": ["V00XRKL"],
            "age": ["V00AGE"], "sex": ["P02SEX"], "bmi": ["P01BMI"],
            "pain": ["V00WOMKPR"], "stiffness": ["V00WOMSTFR"], "function": ["V00WOMADLR"]}
    out_csv = tmp_path / "oai_clinical.csv"
    out_reports = tmp_path / "reports"
    counts = build_dataset(raw, out_csv, out_reports, cols, split_sizes=(0.7, 0.15, 0.15), seed=1)

    assert counts["subjects"] == 30
    assert counts["knee_records"] == 60  # two knees per subject
    assert counts["reports_written"] == 60
    assert all(counts["splits"][s] > 0 for s in ("train", "val", "test"))

    reports = load_reports_from_folders(out_reports)
    assert sum(len(texts) for texts, _ in reports.values()) == 60

    import csv as _csv
    with open(out_csv) as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 60
    assert {r["kl_grade"] for r in rows} == {"0", "1", "2", "3", "4"}

    splits = {r["id"]: r["split"] for r in rows}
    assert len(splits) == 30  # every subject maps to exactly one split (no patient overlap)
    assert set(splits.values()) <= {"train", "val", "test"}
    assert any(r["side"] == "right" and r["kl_grade"] != "" for r in rows)


def test_build_dataset_extracts_radiographic_findings(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    kxr = ["ID|SIDE|V00XRKL|V00XRJSM|V00XRJSL|V00XROSFM|V00XROSTM|V00XROSFL|V00XROSTL"
           "|V00XRSCFM|V00XRSCTM|V00XRSCFL|V00XRSCTL|V00XRATTM|V00XRATTL",
           "00001|1: Right|3: 3|2: 2|0: 0|1: 1|3: 3|0: 0|0: 0|1: 1|2: 2|0: 0|0: 0|2: 2|0: 0"]
    clin = ["ID|V00AGE", "00001|60"]
    (raw / "kxr_sq_bu00.txt").write_text("\n".join(kxr) + "\n")
    (raw / "AllClinical00.txt").write_text("\n".join(clin) + "\n")

    cols = {"id": ["ID"], "side": ["SIDE"], "kl": ["V00XRKL"], "age": ["V00AGE"],
            "jsn_m": ["V00XRJSM"], "jsn_l": ["V00XRJSL"],
            "osteophyte_fm": ["V00XROSFM"], "osteophyte_tm": ["V00XROSTM"],
            "osteophyte_fl": ["V00XROSFL"], "osteophyte_tl": ["V00XROSTL"],
            "sclerosis_fm": ["V00XRSCFM"], "sclerosis_tm": ["V00XRSCTM"],
            "sclerosis_fl": ["V00XRSCFL"], "sclerosis_tl": ["V00XRSCTL"],
            "attrition_m": ["V00XRATTM"], "attrition_l": ["V00XRATTL"]}
    out_csv = tmp_path / "oai_clinical.csv"
    build_dataset(raw, out_csv, tmp_path / "reports", cols)

    import csv as _csv
    with open(out_csv) as f:
        row = next(_csv.DictReader(f))

    assert row["jsn_m"] == "2" and row["jsn_l"] == "0"
    assert row["osteophyte_m"] == "3"  # max(femoral=1, tibial=3)
    assert row["sclerosis_m"] == "2"   # max(femoral=0, tibial=2)
    assert row["attrition_m"] == "2"
    assert "RADIOGRAPHIC FINDINGS" in row["report"]
    assert "moderate joint space narrowing" in row["report"]  # medial JSN=2
    assert "KL" not in row["report"]  # KL grade (3) never appears in the text


def test_build_dataset_missing_columns(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "kxr_sq_bu00.txt").write_text("ID|SIDE|OTHER\n1|1: Right|9\n")
    (raw / "AllClinical00.txt").write_text("ID|V00AGE\n1|62\n")
    cols = {"id": ["ID"], "side": ["SIDE"], "kl": ["V00XRKL"],
            "age": ["V00AGE"], "sex": ["P02SEX"], "bmi": ["P01BMI"],
            "pain": ["V00WOMKPR"], "stiffness": ["V00WOMSTFR"], "function": ["V00WOMADLR"]}
    try:
        build_dataset(raw, tmp_path / "o.csv", tmp_path / "r", cols)
    except ValueError as exc:
        assert "Missing kxr_sq_bu00 columns" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing KL column")
