# KneeVision++

Explainable multimodal AI for knee osteoarthritis severity grading and personalized rehabilitation recommendation.

## Progress

### Phase 1 — Project Setup ✅
- Project structure with modular `src/kneevision/` layout
- Configuration management (`config/settings.py`)
- Dependency management with `uv` / `pyproject.toml`

### Phase 2 — X-ray Analysis (KL Classification) ✅
- **Models:** DenseNet121 & EfficientNet-B4 (with pretrained ImageNet weights) + custom classifier heads
- **Dataset:** Kaggle Knee OA — 8,260 X-rays (KL grades 0-4)
- **Training:** Train/val/test splits, Focal Loss for class imbalance, cosine annealing LR
- **Evaluation:** Ensemble prediction, Cohen's Kappa, Confusion Matrix, Classification Report
- **Scripts:** `scripts/train_xray.py`, `scripts/compare_models.py`, `scripts/evaluate.py`

### Phase 3 — Explainable AI (XAI) ✅
- Abstract Base XAI implementation (`src/kneevision/xai/base.py`)
- Custom Grad-CAM (`src/kneevision/xai/gradcam.py`)
- Custom LIME (`src/kneevision/xai/lime.py`)
- Custom Score-CAM (`src/kneevision/xai/scorecam.py`)
- Heatmap overlay generation

### Upcoming Phases
- Phase 4 — Clinical Information Processing (BioClinicalBERT)
- Phase 5 — Multimodal Fusion + KL Grade Prediction
- Phase 6 — RAG Rehabilitation System
- Phase 7 — FastAPI Backend + React Frontend

## Setup

```bash
uv sync --python 3.14
source .venv/bin/activate
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

## Dataset

- **Kaggle Knee Osteoarthritis Dataset** — 8,260 knee X-rays with KL grades 0-4
- Split: train (5,778), val (826), test (1,656)
