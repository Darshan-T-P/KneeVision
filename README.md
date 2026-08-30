# KneeVision++

Explainable multimodal AI for knee osteoarthritis severity grading and personalized rehabilitation recommendation.

## Architecture

```
src/kneevision/
├── config/settings.py      # env + experiment configuration
├── data/                   # dataset loading (prepare.py), datasets, transforms
├── models/image_model.py   # backbone registry (DenseNet/EfficientNet/ConvNeXt/ViT/Swin)
├── training/               # trainer (EMA, early stopping, AMP), losses
├── evaluation/             # TTA + ensemble prediction
├── xai/                    # Grad-CAM, LIME, Score-CAM
├── clinical/               # BioClinicalBERT text model, clinical dataset, report loaders
├── fusion/                 # (next) multimodal fusion
├── rag/                    # (next) RAG rehab system
└── api/                    # (next) FastAPI backend
```

## Progress

### Phase 1 — Project Setup ✅
- Project structure with modular `src/kneevision/` layout
- Configuration management (`config/settings.py`)
- Dependency management with `uv` / `pyproject.toml`

### Phase 2 — X-ray Analysis (KL Classification) ✅
- **Models:** DenseNet121, EfficientNet-B4, ViT-B/16 (pretrained ImageNet) + custom classifier heads, backed by a model registry
- **Dataset:** Kaggle Knee OA — 8,260 X-rays (KL grades 0-4)
- **Training:** Train/val/test splits, Focal/Ordinal loss, cosine annealing LR, AMP, EMA, early stopping
- **Evaluation:** Ensemble prediction, Cohen's Kappa, Confusion Matrix, Classification Report
- **Scripts:** `scripts/train_xray.py`, `scripts/compare_models.py`, `scripts/evaluate.py`

### Phase 3 — Explainable AI (XAI) ✅
- Abstract Base XAI implementation (`src/kneevision/xai/base.py`)
- Custom Grad-CAM (`src/kneevision/xai/gradcam.py`)
- Custom LIME (`src/kneevision/xai/lime.py`)
- Custom Score-CAM (`src/kneevision/xai/scorecam.py`)
- Heatmap overlay generation

### Code Restructuring ✅
- Single source of truth for dataset loading in `data/prepare.py` (removed 3x duplicated loader)
- Class weights / minority labels derived from data instead of hardcoded counts
- `models/image_model.py` refactored to a backbone registry (add a model = 1 line)
- `load_trained_model()` for robust checkpoint loading (ordinal auto-detection)
- AMP support centralized in `train_epoch`
- Smoke tests added (`tests/`), all pass on CI

### Phase 4 — Clinical Information Processing (BioClinicalBERT) ✅
- `ClinicalTextModel` — BioClinicalBERT encoder + classifier, `extract_features()` ready for fusion
- `ClinicalTextDataset` — tokenized (report, KL grade) pairs
- Report loaders for `{split}/{kl}/*.txt` folders and CSVs
- Synthetic radiology report generator (fallback until real reports are available)
- `scripts/train_clinical.py` — training with Focal loss, early-stop on kappa, MLflow tracking
- **Data sources:**
  - **OAI (primary, free)** — clinical data + gold-standard KL grades, via NIMH Data Archive (NDA) registration. `scripts/download_oai.py` ingests the downloaded tables into the training format. See `data/oai/README.md`.
  - Expert-annotated OAI radiology reports (IEEE DataPort, DOI 10.21227/vcpg-qm58, subscription required) — for real narrative report text.

### Phase 7 — Demo App (Streamlit) ✅
- `streamlit_app.py` — interactive showcase: X-ray KL prediction with confidence chart, Grad-CAM / Score-CAM / LIME heatmaps, BioClinicalBERT clinical-text prediction, model performance dashboard

### Upcoming Phases
- Phase 5 — Multimodal Fusion + KL Grade Prediction
- Phase 6 — RAG Rehabilitation System
- Phase 7b — FastAPI backend + React frontend (production)

## MLflow Tracking & Reports

Every `scripts/evaluate.py` run logs a full evaluation report to MLflow (SQLite, `mlflow.db`):
- **Metrics:** accuracy, quadratic/linear kappa, macro/weighted F1, per-class precision/recall/F1/AUC
- **Artifacts:** `report.html` (self-contained), `confusion_matrix.png`, `classification_report.txt`
- **Model Registry:** single-model runs are registered (e.g. `kneevision_densenet121`, alias `champion`)

Generate a report standalone (metrics + artifacts, no MLflow):
```python
from kneevision.evaluation.report import write_artifacts
write_artifacts(labels, preds, probs, out_dir="reports/evaluate")
```

## Setup

```bash
uv sync --extra dev --python 3.14
```

- `--extra dev` installs pytest, ruff, streamlit, and dvc (needed for tests, linting, and the demo app).
- Optional: set `HF_TOKEN` to avoid unauthenticated Hugging Face Hub warnings and get faster model downloads:
  ```bash
  export HF_TOKEN=hf_your_token
  ```
- If `uv sync` ever re-upgrades MLflow and Python 3.14 import errors appear, re-apply the compatibility patch:
  ```bash
  uv run python scripts/fix_mlflow_py314.py
  ```

## Commands

All commands run through `uv run` so they use the project's virtualenv. Scripts expect the dataset at `data/raw/{train,val,test}/{kl_grade}/*.png` and checkpoints in `models/`.

### Train

Train the default X-ray model (DenseNet121, train/val):
```bash
uv run python scripts/train_xray.py
```

Train and compare DenseNet121 vs EfficientNet-B4 (Focal loss, EMA, AMP, logs to MLflow):
```bash
uv run python scripts/compare_models.py
```

Train the clinical-text model (BioClinicalBERT). Without real reports, generate a synthetic dataset first:
```bash
uv run python scripts/train_clinical.py --synthetic --epochs 5
```

Or train on real report folders (`data/clinical/{train,val}/{kl_grade}/*.txt`):
```bash
uv run python scripts/train_clinical.py --data data/clinical --epochs 10
```

Extra `train_clinical.py` flags: `--image-data <dir>` (KL grades derived from image splits), `--batch-size`, `--lr`, `--freeze` (freeze the encoder), `--num-classes`.

### OAI clinical dataset (NDA)

Real clinical data (KL-grade labels + demographics/WOMAC) from the Osteoarthritis
Initiative — ingested and ready at `data/oai/processed/oai_clinical.csv`:

| Split | KL 0 | KL 1 | KL 2 | KL 3 | KL 4 | Total |
|-------|-----:|-----:|-----:|-----:|-----:|------:|
| train | 5,599 | 2,494 | 2,190 | 1,061 | 230 | 11,574 |
| val   | 1,273 |   558 |   480 |   196 |  33 |  2,540 |
| test  | 1,213 |   523 |   482 |   209 |  51 |  2,478 |

Subjects are never split across train/val/test. Access is free but requires NDA
registration/approval — full steps in `data/oai/README.md`:

```bash
uv run python scripts/download_oai.py status              # access steps + which raw files are present
uv run python scripts/download_oai.py ingest              # build dataset (after files are in data/oai/raw/)
uv run python scripts/train_clinical.py --data data/oai/processed/oai_clinical.csv --epochs 10
```

### Evaluate

Evaluate a single model on the test set (registers `kneevision_<name>` in the MLflow Model Registry with alias `champion`):
```bash
uv run python scripts/evaluate.py densenet121
```

Evaluate an ensemble on the test set (logs metrics + artifacts, no registry):
```bash
uv run python scripts/evaluate.py densenet121 vit_b_16
```

Stale/incompatible checkpoints are skipped automatically with a warning. Output includes accuracy, quadratic kappa, per-class precision/recall/F1, and a normalized confusion matrix.

### Demo app

Interactive Streamlit showcase — X-ray diagnosis, Grad-CAM/Score-CAM/LIME explainability, clinical-text prediction, model performance:
```bash
uv run --extra dev streamlit run streamlit_app.py
# open http://localhost:8501
```

### MLflow UI

View experiment runs, metrics, artifacts, and registered models:
```bash
uv run python scripts/mlflow_server.py server --port 5000
# open http://localhost:5000
```

### Tests & lint

```bash
uv run pytest -q     # 35 unit tests
uv run ruff check src scripts streamlit_app.py tests
```

## Dataset

**Kaggle Knee Osteoarthritis Dataset** — 8,260 knee X-rays with KL grades 0–4
(source zip: `data/archive.zip`, extracted to `data/raw/{train,val,test}/{kl_grade}/*.png`;
an extra auto-labeled `auto_test/` folder in the zip is not used).

Split (verified on disk):

| Split | KL 0 | KL 1 | KL 2 | KL 3 | KL 4 | Total |
|-------|-----:|-----:|-----:|-----:|-----:|------:|
| train | 2,286 | 1,046 | 1,516 | 757 | 173 | 5,778 |
| val   |   328 |   153 |   212 | 106 |  27 |   826 |
| test  |   639 |   296 |   447 | 223 |  51 | 1,656 |

The dataset is imbalanced (KL4 ≈ 3% of training images); this is handled at
training time, not by offline oversampling — no augmented copies are stored on disk.

## Augmentation (on-the-fly only)

Applied per-batch during training via `src/kneevision/data/transforms.py` and `dataset.py`:

- **Standard train pipeline** (`train_transform`): resize 256 → RandomResizedCrop(224, scale 0.8–1.0) → horizontal flip p=0.5 → rotation ±15° → brightness/contrast jitter ±0.2 → RandAugment (2 ops, magnitude 9) → random translate ±5% → sharpness p=0.3 → Gaussian blur → RandomErasing p=0.25 → ImageNet normalize.
- **Minority-class pipeline** (`minority_transform`): stronger version applied to under-represented grades (auto-detected as < 40% of the largest class — currently KL3/KL4): larger crop jitter (0.7–1.0), rotation ±25°, stronger color jitter, affine with shear/scale, blur up to σ=1.0.
- **MixUp / CutMix** (`MixUpDataset`, α=0.4): 50% of samples get either beta-mixed pixels or a 10–30% cut-paste patch, labels become soft one-hots.
- **Class-balanced sampling** (`make_weighted_sampler`, power=0.5): inverse-sqrt-frequency weighted sampler so rare grades are seen more often per epoch.
- **Val/test**: deterministic resize 224 + normalize only. **TTA**: base + horizontal flip (`TTA_AUGS = 2`).
