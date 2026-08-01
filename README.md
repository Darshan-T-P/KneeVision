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
- **Data sources:** OAI (X-rays + clinical scores, https://nda.nih.gov/oai) and expert-annotated OAI radiology reports (IEEE DataPort, DOI 10.21227/vcpg-qm58)

### Phase 7 — Demo App (Streamlit) ✅
- `streamlit_app.py` — interactive showcase: X-ray KL prediction with confidence chart, Grad-CAM / Score-CAM / LIME heatmaps, BioClinicalBERT clinical-text prediction, model performance dashboard

### Upcoming Phases
- Phase 5 — Multimodal Fusion + KL Grade Prediction
- Phase 6 — RAG Rehabilitation System
- Phase 7b — FastAPI backend + React frontend (production)

## Demo App

Interactive Streamlit showcase — X-ray diagnosis, Grad-CAM/Score-CAM/LIME explainability, clinical text prediction, and model performance:

```bash
uv run --extra dev streamlit run streamlit_app.py
```

## MLflow Tracking & Reports

Every `scripts/evaluate.py` run logs a full evaluation report to MLflow (SQLite, `mlflow.db`):
- **Metrics:** accuracy, quadratic/linear kappa, macro/weighted F1, per-class precision/recall/F1/AUC
- **Artifacts:** `report.html` (self-contained), `confusion_matrix.png`, `classification_report.txt`
- **Model Registry:** single-model runs are registered (e.g. `kneevision_densenet121`, alias `champion`)

Launch the UI:
```bash
uv run python scripts/mlflow_server.py server --port 5000
# open http://localhost:5000
```

Generate a report standalone (metrics + artifacts, no MLflow):
```python
from kneevision.evaluation.report import write_artifacts
write_artifacts(labels, preds, probs, out_dir="reports/evaluate")
```

## Setup

```bash
uv sync --python 3.14
```

## Usage

Train and compare the X-ray classifiers (DenseNet121 & EfficientNet-B4):
```bash
python scripts/compare_models.py
```

Evaluate an ensemble of the trained models on the test set:
```bash
python scripts/evaluate.py
```

Run tests:
```bash
uv run pytest
```

## Dataset

- **Kaggle Knee Osteoarthritis Dataset** — 8,260 knee X-rays with KL grades 0-4
- Split: train (5,778), val (826), test (1,656)
