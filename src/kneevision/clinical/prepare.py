import random
from pathlib import Path
import csv

TEXT_EXTENSIONS = {".txt", ".md"}


def load_reports_from_folders(root: Path) -> dict[str, tuple[list[str], list[int]]]:
    """Load reports organized as: root/{split}/{kl_grade}/*.txt"""
    splits = {}
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        texts, labels = [], []
        for grade_dir in sorted(split_dir.iterdir()):
            if not grade_dir.is_dir():
                continue
            label = int(grade_dir.name)
            for report_path in sorted(grade_dir.glob("*.*")):
                if report_path.suffix.lower() in TEXT_EXTENSIONS:
                    texts.append(report_path.read_text())
                    labels.append(label)
        if texts:
            splits[split] = (texts, labels)
    return splits


def load_reports_csv(csv_path: Path, text_col: str = "report",
                     label_col: str = "kl_grade") -> dict[str, tuple[list[str], list[int]]]:
    """Load reports from a CSV with a `split` column.

    Expects columns: {text_col}, {label_col}, split
    """
    splits: dict[str, tuple[list[str], list[int]]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "train").strip()
            if text_col not in row or label_col not in row:
                continue
            texts, labels = splits.setdefault(split, ([], []))
            texts.append(row[text_col])
            labels.append(int(row[label_col]))
    return splits


# --- Synthetic report generation (fallback until real reports are available) ---

_JSN = {
    0: "No evidence of joint space narrowing",
    1: "Minimal joint space narrowing",
    2: "Mild joint space narrowing",
    3: "Moderate joint space narrowing",
    4: "Severe joint space narrowing with near-complete loss of the medial joint space",
}

_OSTEOPHYTES = {
    0: "No osteophytes identified",
    1: "Possible small osteophytes",
    2: "Definite small osteophytes at the medial and/or lateral margins",
    3: "Multiple moderate osteophytes",
    4: "Large osteophytes involving multiple compartments",
}

_SCLEROSIS = {
    0: "No subchondral sclerosis",
    1: "No significant subchondral sclerosis",
    2: "Mild subchondral sclerosis",
    3: "Moderate subchondral sclerosis",
    4: "Marked subchondral sclerosis",
}

_IMPRESSION = {
    0: "No radiographic evidence of osteoarthritis. KL grade 0.",
    1: "Doubtful osteoarthritis. KL grade 1.",
    2: "Mild osteoarthritis. KL grade 2.",
    3: "Moderate osteoarthritis. KL grade 3.",
    4: "Severe osteoarthritis. KL grade 4.",
}


def generate_report(kl_grade: int, seed: int = 42) -> str:
    """Generate a radiology-style report consistent with a KL grade."""
    rng = random.Random(f"{seed}-{kl_grade}")
    laterality = rng.choice(["right", "left"])
    opening = rng.choice([
        f"AP and lateral radiographs of the {laterality} knee were reviewed.",
        f"Bilateral AP standing radiographs were reviewed, {laterality} knee is reported.",
        f"Radiographs of the {laterality} knee obtained today were compared with prior films.",
    ])
    soft = ""
    if kl_grade >= 2:
        soft = f" {rng.choice(['Chondrocalcinosis is not evident.', 'No intra-articular bodies identified.', 'Joint effusion is not present.'])}"

    return (
        f"{opening}\n"
        f"FINDINGS: {_JSN[kl_grade]}. {_OSTEOPHYTES[kl_grade]}. {_SCLEROSIS[kl_grade]}.{soft}\n"
        f"IMPRESSION: {_IMPRESSION[kl_grade]}"
    )


def generate_synthetic_dataset(labels: list[int], out_dir: Path, seed: int = 42) -> list[Path]:
    """Write one generated report per label into out_dir/{kl_grade}/report_{i}.txt."""
    written = []
    for i, kl_grade in enumerate(labels):
        dest = out_dir / str(kl_grade) / f"report_{i:05d}.txt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(generate_report(kl_grade, seed))
        written.append(dest)
    return written


def load_reports_from_grades(root: Path) -> tuple[list[str], list[int]]:
    """Load reports organized as: root/{kl_grade}/*.txt (no split subdirectory)."""
    texts, labels = [], []
    for grade_dir in sorted(root.iterdir()):
        if not grade_dir.is_dir():
            continue
        label = int(grade_dir.name)
        for report_path in sorted(grade_dir.glob("*.*")):
            if report_path.suffix.lower() in TEXT_EXTENSIONS:
                texts.append(report_path.read_text())
                labels.append(label)
    return texts, labels


def build_synthetic_splits(image_labels_by_split: dict[str, tuple[list, list[int]]],
                           out_root: Path, seed: int = 42) -> dict[str, tuple[list[str], list[int]]]:
    """Mirror the image split structure as synthetic clinical reports."""
    splits = {}
    for split, (_, labels) in image_labels_by_split.items():
        dest_dir = out_root / split
        generate_synthetic_dataset(labels, dest_dir, seed=seed)
        splits[split] = load_reports_from_grades(dest_dir)
    return splits
