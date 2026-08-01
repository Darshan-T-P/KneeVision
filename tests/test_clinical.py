
from kneevision.clinical.prepare import (
    generate_report,
    generate_synthetic_dataset,
    load_reports_from_grades,
    load_reports_from_folders,
    load_reports_csv,
    build_synthetic_splits,
)


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
