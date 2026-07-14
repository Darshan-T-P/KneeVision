# KneeVision++

Explainable multimodal AI for knee osteoarthritis severity grading and personalized rehabilitation recommendation.

## Progress

### Phase 1 — Project Setup ✅
- Project structure with modular `src/kneevision/` layout
- Configuration management (`config/settings.py`)
- Dependency management with `uv` / `pyproject.toml`

### Phase 2 — X-ray Analysis (KL Classification) ✅
- **Model:** DenseNet121 (with pretrained ImageNet weights) + custom classifier head
- **Dataset:** Kaggle Knee OA — 8,260 X-rays (KL grades 0-4)
- **Training:** Train/val split, weighted loss for class imbalance, cosine annealing LR
- **Script:** `scripts/train_xray.py`

### Phase 3 — Explainable AI (Grad-CAM) ✅
- Custom Grad-CAM implementation (no extra dependencies)
- Heatmap overlay generation
- `src/kneevision/xai/gradcam.py`

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

Train the X-ray classifier:
```bash
python scripts/train_xray.py
```

## Dataset

- **Kaggle Knee Osteoarthritis Dataset** — 8,260 knee X-rays with KL grades 0-4
- Split: train (5,778), val (826), test (1,656)
