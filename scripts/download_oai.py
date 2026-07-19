import argparse
import csv
import shutil
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kneevision.data.prepare import print_split_summary
from kneevision.utils.logging import setup_logger

logger = setup_logger("download_oai")


def parse_kl_assessments(csv_path: Path) -> dict[str, int]:
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
                subj_id = None
                for sk in ("ID", "SUBJECT_ID", "Subject ID", "USUBJID", "subjectid"):
                    if sk in row:
                        subj_id = row[sk]
                        break
                if subj_id:
                    kl_map[subj_id] = int(kl)
    return kl_map


def match_images_to_labels(image_dir: Path, kl_map: dict[str, int]) -> list[tuple[Path, int]]:
    matched = []
    for fpath in sorted(image_dir.rglob("*.dcm")) + sorted(image_dir.rglob("*.png")):
        for subj_id, kl in kl_map.items():
            if subj_id in fpath.stem:
                matched.append((fpath, kl))
                break
    return matched


def organize_splits(matched: list[tuple[Path, int]], output_dir: Path,
                    split_ratio: tuple = (0.7, 0.15, 0.15), seed: int = 42):
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

    return {split: ([p for p, _ in items], [label for _, label in items])
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
        logger.error("Image directory not found: %s", image_dir)
        logger.info("\nDownload images first via NDA Download Tool:")
        logger.info("  1. Go to https://nda.nih.gov/oai")
        logger.info("  2. Data Download -> Select 'OAI Fixed-Flexion Knee X-rays'")
        logger.info("  3. Generate manifest -> Download with nda-tools")
        return

    if not assessment_path.exists():
        logger.error("Assessment file not found: %s", assessment_path)
        logger.info("\nDownload from NDA:")
        logger.info("  1. Data Download -> Select 'Knee X-Ray Image Assessments - ASCII'")
        logger.info("  2. Unzip and point --assessments to the CSV file")
        return

    logger.info("Loading KL assessments from %s ...", assessment_path)
    kl_map = parse_kl_assessments(assessment_path)
    logger.info("  Found %d KL grade entries", len(kl_map))

    logger.info("Matching images from %s ...", image_dir)
    matched = match_images_to_labels(image_dir, kl_map)
    logger.info("  Matched %d images to KL grades", len(matched))

    if not matched:
        logger.warning("\nNo matches found. Check the assessment CSV column names.")
        logger.warning("Open the CSV and look for the subject ID and KL grade columns.")
        logger.warning("Then update parse_kl_assessments() in this script.")
        return

    logger.info("Organizing into %s ...", output_dir)
    splits = organize_splits(matched, output_dir)
    logger.info("Done! Split summary:")
    print_split_summary(splits)


if __name__ == "__main__":
    main()