from pathlib import Path
import csv
import random

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def get_paths_and_labels(split_dir: Path) -> tuple[list[Path], list[int]]:
    """Load a split organized as: split_dir/{kl_grade}/*.png"""
    paths, labels = [], []
    for grade_dir in sorted(split_dir.iterdir()):
        if not grade_dir.is_dir():
            continue
        label = int(grade_dir.name)
        for img_path in sorted(grade_dir.glob("*.*")):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(img_path)
                labels.append(label)
    return paths, labels


def get_splits(raw_dir: Path) -> dict[str, tuple[list[Path], list[int]]]:
    """Load all splits (train/val/test) from raw_dir/{split}/{kl_grade}/."""
    splits = {}
    for split in ["train", "val", "test"]:
        split_dir = raw_dir / split
        if split_dir.exists():
            splits[split] = get_paths_and_labels(split_dir)
    return splits


def class_distribution(labels: list[int], num_classes: int = 5) -> list[int]:
    """Count label frequencies, returning a fixed-length list."""
    counts = [0] * num_classes
    for label in labels:
        counts[label] += 1
    return counts


def class_weights(labels: list[int], num_classes: int = 5) -> list[float]:
    """Inverse-frequency class weights normalized to sum to num_classes."""
    counts = class_distribution(labels, num_classes)
    total = sum(counts)
    return [(total / (c * num_classes)) if c > 0 else 0.0 for c in counts]


def minority_labels(labels: list[int], num_classes: int = 5, threshold: float = 0.4) -> set[int]:
    """Classes whose frequency is below `threshold * max_count` (data-driven)."""
    counts = class_distribution(labels, num_classes)
    max_count = max(counts)
    return {i for i, c in enumerate(counts) if c < threshold * max_count}


def prepare_from_folders(raw_dir: Path) -> dict[str, tuple[list[Path], list[int]]]:
    """Load dataset organized as: raw_dir/{split}/{kl_grade}/*.png"""
    return get_splits(raw_dir)


def prepare_from_csv(csv_path: Path, image_root: Path) -> dict[str, tuple[list[Path], list[int]]]:
    """Load dataset from a CSV with columns: filename, label, split
    Example: 9006401.png,2,train"""
    splits: dict[str, tuple[list[Path], list[int]]] = {"train": ([], []), "val": ([], []), "test": ([], [])}
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            filename, label, split = row[0], int(row[1]), row[2].strip()
            if split not in splits:
                continue
            img_path = image_root / filename
            if img_path.exists():
                splits[split][0].append(img_path)
                splits[split][1].append(label)
    return splits


def prepare_oai(oai_root: Path, split_ratio: tuple = (0.7, 0.15, 0.15), seed: int = 42):
    """Prepare OAI dataset from downloaded OAI X-ray images.
    Expects: oai_root/ contains subdirectories with KL-graded X-rays.
    OAI central readings are at: oai_root/Enrollees/*/CentralRead/KXR*.xml
    """
    random.seed(seed)
    all_images = list(oai_root.rglob("*.png")) + list(oai_root.rglob("*.jpg"))
    if not all_images:
        print(f"No images found in {oai_root}. Place OAI X-rays here.")
        return {}

    paths, labels = [], []
    print(f"Found {len(all_images)} images. Parsing OAI directory structure...")
    for img_path in sorted(all_images):
        label = _infer_oai_label(img_path)
        if label is not None:
            paths.append(img_path)
            labels.append(label)

    if not paths:
        print("Could not infer KL grades from folder structure.")
        print("Expected format: OAI CentralRead XML files alongside images.")
        print("Manual: place images in data/raw/train/{kl}/ and data/raw/test/{kl}/")
        return {}

    combined = list(zip(paths, labels))
    random.shuffle(combined)
    n = len(combined)
    t1, t2 = int(n * split_ratio[0]), int(n * (split_ratio[0] + split_ratio[1]))
    return {
        "train": (list(list(zip(*combined[:t1]))[0]), list(list(zip(*combined[:t1]))[1])),
        "val":   (list(list(zip(*combined[t1:t2]))[0]), list(list(zip(*combined[t1:t2]))[1])),
        "test":  (list(list(zip(*combined[t2:]))[0]),   list(list(zip(*combined[t2:]))[1])),
    }


def _infer_oai_label(img_path: Path) -> int | None:
    """Try to extract KL grade from OAI filename or parent directory."""
    try:
        parent = img_path.parent.name
        if parent.startswith("KL") or parent.startswith("kl"):
            return int(parent[2:])
        parts = img_path.stem.split("_")
        for p in parts:
            if p.startswith("KL") or p.startswith("kl"):
                return int(p[2:])
    except (ValueError, IndexError):
        pass
    return None


def print_split_summary(splits: dict):
    for split, (paths, labels) in splits.items():
        if not paths:
            continue
        dist = [labels.count(i) for i in range(max(labels) + 1)]
        print(f"{split:6s}: {len(paths):5d} images | distribution: {dist}")
