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
                     label_col: str = "kl_grade", return_features: bool = False) -> dict:
    """Load reports from a CSV with a `split` column.
    If return_features is True, returns dict of {split: (texts, labels, features)}.
    Otherwise returns {split: (texts, labels)}.
    """
    splits = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "train").strip()
            if text_col not in row or label_col not in row:
                continue
                
            if split not in splits:
                if return_features:
                    splits[split] = ([], [], [])
                else:
                    splits[split] = ([], [])
                    
            splits[split][0].append(row[text_col])
            splits[split][1].append(int(row[label_col]))
            if return_features:
                feature_keys = ["age", "sex", "bmi", "pain", "stiffness", "function",
                                "injury", "surgery", "meds"] + RADIOGRAPHIC_FIELDS
                features = {k: row[k] for k in feature_keys if k in row}
                splits[split][2].append(features)
    return splits


# --- Real OARSI atlas per-compartment finding phrases (0-3 grade scale) ---
# These come from the OAI's kxr_sq_bu00.txt semi-quantitative reads and are
# independent per-compartment radiographic gradings, not the KL grade itself.

RADIOGRAPHIC_FIELDS = [
    "jsn_m", "jsn_l",
    "osteophyte_m", "osteophyte_l",
    "sclerosis_m", "sclerosis_l",
    "attrition_m", "attrition_l",
]

_REAL_JSN = {
    0: "no joint space narrowing",
    1: "mild joint space narrowing",
    2: "moderate joint space narrowing",
    3: "severe joint space narrowing",
}

_REAL_OSTEOPHYTES = {
    0: "no osteophytes",
    1: "small osteophytes",
    2: "definite osteophytes",
    3: "large osteophytes",
}

_REAL_SCLEROSIS = {
    0: "no subchondral sclerosis",
    1: "mild subchondral sclerosis",
    2: "moderate subchondral sclerosis",
    3: "marked subchondral sclerosis",
}

_REAL_ATTRITION = {
    1: "mild attritional bone change",
    2: "moderate attritional bone change",
    3: "severe attritional (bone-on-bone) change",
}


def _grade_int(value) -> int | None:
    """Parse a per-compartment grade that may arrive as int, float-string, or
    an empty/missing CSV cell (""), returning None if it can't be read."""
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _compartment_findings(jsn, osteophyte, sclerosis) -> str | None:
    jsn, osteophyte, sclerosis = _grade_int(jsn), _grade_int(osteophyte), _grade_int(sclerosis)
    bits = []
    if jsn is not None:
        bits.append(_REAL_JSN.get(jsn, "joint space narrowing not assessed"))
    if osteophyte is not None:
        bits.append(_REAL_OSTEOPHYTES.get(osteophyte, "osteophytes not assessed"))
    if sclerosis is not None:
        bits.append(_REAL_SCLEROSIS.get(sclerosis, "sclerosis not assessed"))
    return ", ".join(bits) if bits else None


def compose_radiographic_findings(features: dict) -> str:
    """Compose a 'RADIOGRAPHIC FINDINGS' sentence from real per-compartment
    OARSI grades (medial/lateral joint space narrowing, osteophytes,
    subchondral sclerosis, attrition) if present in `features`. Returns an
    empty string if none of these fields are available. Never includes the
    overall KL grade — only its independent structural components.
    """
    parts = []
    medial = _compartment_findings(
        features.get("jsn_m"), features.get("osteophyte_m"), features.get("sclerosis_m")
    )
    if medial:
        parts.append(f"Medial compartment: {medial}.")

    lateral = _compartment_findings(
        features.get("jsn_l"), features.get("osteophyte_l"), features.get("sclerosis_l")
    )
    if lateral:
        parts.append(f"Lateral compartment: {lateral}.")

    for side_label, key in (("medially", "attrition_m"), ("laterally", "attrition_l")):
        value = _grade_int(features.get(key))
        if value is not None and value > 0:
            phrase = _REAL_ATTRITION.get(value)
            if phrase:
                parts.append(f"{phrase.capitalize()} is noted {side_label}.")

    if not parts:
        return ""
    return "RADIOGRAPHIC FINDINGS: " + " ".join(parts)


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


def compose_clinical_report(features: dict, side: str = "right", augment: bool = False) -> str:
    """Compose a structured clinical summary from OAI-style baseline features.

    `features` may contain: age, sex, bmi, WOMAC 0-100 symptom items `pain`,
    `stiffness`, `function` (or a legacy `womac_total`), and real per-compartment
    OARSI radiographic grades `jsn_m`/`jsn_l`, `osteophyte_m`/`osteophyte_l`,
    `sclerosis_m`/`sclerosis_l`, `attrition_m`/`attrition_l` (0-3 scale, from
    kxr_sq_bu00.txt). The overall KL grade is deliberately NOT included in the
    text: it is the label the model must predict, and embedding it would leak
    the answer — the radiographic fields above are its independent structural
    components, not the grade itself. Missing fields are simply omitted.
    """
    def _fmt(value):
        if value is None:
            return None
        try:
            return f"{float(value):g}"
        except (TypeError, ValueError):
            return str(value)

    age, sex, bmi = (features.get(k) for k in ("age", "sex", "bmi"))

    parts = [f"Baseline clinical assessment of the {side} knee."]

    findings = compose_radiographic_findings(features)
    if findings:
        parts.append(findings)

    demos = []
    if age is not None:
        demos.append(f"{_fmt(age)} years old")
    if sex:
        demos.append(str(sex))
    if bmi is not None:
        demos.append(f"BMI {_fmt(bmi)}")
    if demos:
        parts.append("Patient is " + ", ".join(demos) + ".")
    if features.get("bmi"):
        parts.append(f"BMI is {_fmt(features['bmi'])}.")
        
    injury = str(features.get("injury", "")).strip()
    if injury.startswith("1"):
        parts.append("Patient has a history of knee injury.")
    elif injury.startswith("0") and not augment:
        parts.append("No prior knee injury reported.")
        
    surgery = str(features.get("surgery", "")).strip()
    if surgery.startswith("1"):
        parts.append("Patient has undergone previous knee surgery.")
    elif surgery.startswith("0") and not augment:
        parts.append("No prior knee surgery.")
        
    meds = str(features.get("meds", "")).strip()
    if meds.startswith("1"):
        parts.append("Patient is currently taking medication for knee pain.")
    elif meds.startswith("0") and not augment:
        parts.append("Patient is not currently on knee pain medication.")

    symptoms = []
    if "womac_total" in features and features["womac_total"] is not None:
        symptoms.append(("overall WOMAC", features["womac_total"]))
    for label, key in (("pain", "pain"), ("stiffness", "stiffness"), ("function", "function")):
        if features.get(key) is not None:
            symptoms.append((f"WOMAC {label}", features[key]))

    for label, value in symptoms:
        try:
            score = float(value)
            severity = "minimal" if score < 24 else "mild to moderate" if score < 48 else "substantial"
            if augment and score == 0.0 and random.random() < 0.5:
                continue # Randomly drop perfect scores to prevent model memorization
        except (TypeError, ValueError):
            score, severity = value, "reported"
            if augment and random.random() < 0.3:
                continue # Randomly drop missing data text

        if augment and random.random() < 0.5:
            synonym = random.choice(["notable", "significant", "substantial"]) if severity == "substantial" else severity
            parts.append(f"Patient reported a {label} score of {_fmt(score)}/100 ({synonym}).")
        else:
            parts.append(
                f"Patient-reported {label} score is {_fmt(score)} of 100, "
                f"indicating {severity} {label.split()[-1]}."
            )
            
    if augment:
        # Keep opening sentence fixed, shuffle the demographic and symptom sentences
        rest = parts[1:]
        random.shuffle(rest)
        parts = parts[:1] + rest
        
    return " ".join(parts)


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
