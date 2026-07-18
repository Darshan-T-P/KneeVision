"""
OAI Dataset Download & Preparation Script

1. Install NDA tools:   pip install ndatools
2. Run:                 nda-tools downloads -d <manifest>
3. Then run this script to organize into train/val/test splits.

Prerequisites:
  - Approved OAI DAR (you have one: ID 25595)
  - Downloaded OAI X-ray images via NDA Download Tool
  - Downloaded Knee X-Ray Image Assessments (ASCII) zip

Usage:
  python scripts/download_oai.py --image_dir /path/to/downloaded/images \\
                                  --assessments /path/to/XXKXR??.csv \\
                                  --output data/raw
"""

import argparse
import csv
import shutil
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kneevision.data.prepare import print_split_summary


def parse_kl_assessments(csv_path: Path) -> dict[str, int]:
    """Parse OAI X-Ray Assessment CSV to get KL grades.
    Columns vary by assessment file version — look for subject ID and KL columns."""
    kl_map = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                kl = None
                if "kl" in key.lower() and ("grade" in key.lower() or "kr" in key.lower()):
                    kl = row[key]
                elif key.strip().upper() in ("V00KXRKL", "V00KL", "KXRKL", "KL"):
                    kl = row[key]
            if kl and kl.isdigit():
                # Subject ID column varies — try common names
                subj_id = None
                for sk in ("ID", "SUBJECT_ID", "Subject ID", "USUBJID", "subjectid"):
                    if sk in row:
                        subj_id = row[sk]
                        break
                if subj_id:
                    kl_map[subj_id] = int(kl)
    return kl_map


def match_images_to_labels(image_dir: Path, kl_map: dict[str, int]) -> list[tuple[Path, int]]:
    """Match DICOM image files to KL grades via subject ID in filename."""
    matched = []
    for fpath in sorted(image_dir.rglob("*.dcm")) + sorted(image_dir.rglob("*.png")):
        # OAI filenames typically contain subject ID
        for subj_id, kl in kl_map.items():
            if subj_id in fpath.stem:
                matched.append((fpath, kl))
                break
    return matched


def organize_splits(matched: list[tuple[Path, int]], output_dir: Path,
                    split_ratio: tuple = (0.7, 0.15, 0.15), seed: int = 42):
    """Copy images into train/val/test organized by KL grade."""
    random.seed(seed)
    random.shuffle(matched)
    n = len(matched)
    t1, t2 = int(n * split_ratio[0]), int(n * (split_ratio[0] + split_ratio[1]))

    splits = {
        "train": matched[:t1],
        "val":   matched[t1:t2],
        "test":  matched[t2:],
    }

    for split_name, items in splits.items():
        for img_path, label in items:
            dest = output_dir / split_name / str(label) / img_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest)

    return {split: ([p for p, _ in items], [l for _, l in items])
            for split, items in splits.items()}


def main():
    parser = argparse.ArgumentParser(description="Prepare OAI dataset")
    parser.add_argument("--image_dir", required=True, help="Directory with OAI X-ray images")
    parser.add_argument("--assessments", required=True, help="CSV from Knee X-Ray Image Assessments")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    assessment_path = Path(args.assessments)
    output_dir = Path(args.output)

    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
        print("\nDownload images first via NDA Download Tool:")
        print("  1. Go to https://nda.nih.gov/oai")
        print("  2. Data Download → Select 'OAI Fixed-Flexion Knee X-rays'")
        print("  3. Generate manifest → Download with nda-tools")
        return

    if not assessment_path.exists():
        print(f"Assessment file not found: {assessment_path}")
        print("\nDownload from NDA:")
        print("  1. Data Download → Select 'Knee X-Ray Image Assessments - ASCII'")
        print("  2. Unzip and point --assessments to the CSV file")
        return

    print(f"Loading KL assessments from {assessment_path} ...")
    kl_map = parse_kl_assessments(assessment_path)
    print(f"  Found {len(kl_map)} KL grade entries")

    print(f"Matching images from {image_dir} ...")
    matched = match_images_to_labels(image_dir, kl_map)
    print(f"  Matched {len(matched)} images to KL grades")

    if not matched:
        print("\nNo matches found. Check the assessment CSV column names.")
        print("Open the CSV and look for the subject ID and KL grade columns.")
        print("Then update parse_kl_assessments() in this script.")
        return

    print(f"Organizing into {output_dir} ...")
    splits = organize_splits(matched, output_dir)
    print("\nDone! Split summary:")
    print_split_summary(splits)


if __name__ == "__main__":
    main()
