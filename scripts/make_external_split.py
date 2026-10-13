"""One-off fixed development/test split of the external Gornale cohort (Expert-I labels).

30% DEV: every choice (adaptation settings, recalibration shifts) may use it.
70% TEST: held out; reported once. Stratified by KL grade, seed 42. The cohort has no patient IDs, so
images of one patient may fall on both sides (results can be slightly optimistic) -- disclose this.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external/gornale/Digital Knee X-ray Images/Knee X-ray Images/expert_I/MedicalExpert-I"
GRADE = {"0Normal": 0, "1Doubtful": 1, "2Mild": 2, "3Moderate": 3, "4Severe": 4}
rng = random.Random(42)
split = {"dev": [], "test": []}
for folder, g in GRADE.items():
    files = sorted((BASE / folder).glob("*.png"))
    rng.shuffle(files)
    n_dev = round(0.30 * len(files))
    split["dev"] += [(str(f.relative_to(ROOT)), g) for f in files[:n_dev]]
    split["test"] += [(str(f.relative_to(ROOT)), g) for f in files[n_dev:]]
out = ROOT / "reports" / "external_gornale_split.json"
out.write_text(json.dumps(split, indent=1))
for k, v in split.items():
    print(k, len(v), [sum(1 for _, g in v if g == i) for i in range(5)])
