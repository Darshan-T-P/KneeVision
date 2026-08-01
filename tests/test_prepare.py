from pathlib import Path
import csv

from kneevision.data.prepare import (
    get_paths_and_labels,
    get_splits,
    prepare_from_folders,
    prepare_from_csv,
    class_distribution,
    class_weights,
    minority_labels,
)


def _make_tree(root: Path):
    (root / "train" / "0").mkdir(parents=True)
    (root / "train" / "1").mkdir(parents=True)
    (root / "train" / "4").mkdir(parents=True)
    (root / "val" / "0").mkdir(parents=True)
    (root / "val" / "2").mkdir(parents=True)
    for path, n in [
        (root / "train" / "0", 10),
        (root / "train" / "1", 5),
        (root / "train" / "4", 2),
        (root / "val" / "0", 3),
        (root / "val" / "2", 1),
    ]:
        for i in range(n):
            (path / f"{i}.png").write_bytes(b"\x89PNG")


def test_get_paths_and_labels(tmp_path):
    _make_tree(tmp_path)
    paths, labels = get_paths_and_labels(tmp_path / "train")
    assert len(paths) == 17
    assert sorted(set(labels)) == [0, 1, 4]
    assert sum(label == 0 for label in labels) == 10


def test_get_splits(tmp_path):
    _make_tree(tmp_path)
    splits = get_splits(tmp_path)
    assert set(splits) == {"train", "val"}
    assert len(splits["train"][0]) == 17
    assert len(splits["val"][0]) == 4


def test_prepare_from_folders_matches_get_splits(tmp_path):
    _make_tree(tmp_path)
    assert prepare_from_folders(tmp_path) == get_splits(tmp_path)


def test_prepare_from_csv(tmp_path):
    _make_tree(tmp_path)
    csv_path = tmp_path / "splits.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "split"])
        writer.writerow(["train/0/0.png", 0, "train"])
        writer.writerow(["missing.png", 2, "train"])
        writer.writerow(["bad_row"])
    splits = prepare_from_csv(csv_path, tmp_path)
    assert len(splits["train"][0]) == 1


def test_class_distribution():
    labels = [0, 0, 0, 1, 1, 4]
    assert class_distribution(labels) == [3, 2, 0, 0, 1]


def test_class_weights():
    labels = [0] * 9 + [1] * 1
    weights = class_weights(labels)
    assert len(weights) == 5
    assert abs(weights[0] - 10 / (9 * 5)) < 1e-6
    assert abs(weights[1] - 2.0) < 1e-6
    assert weights[0] < weights[1]
    assert weights[2] == 0.0


def test_minority_labels():
    labels = [0] * 100 + [1] * 40 + [3] * 30 + [4] * 10
    assert minority_labels(labels, threshold=0.4) == {2, 3, 4}
