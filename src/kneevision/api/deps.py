"""Lazy, cached model loaders for the API — mirrors the @st.cache_resource
pattern in streamlit_app.py: load once on first request, reuse after."""
from functools import lru_cache

from kneevision.config.settings import MODELS_DIR, GUIDELINES_DIR
from kneevision.utils.helpers import get_device
from kneevision.models.image_model import load_trained_model, KneeXRayClassifier, AVAILABLE_MODELS
from kneevision.clinical.model import load_trained_clinical_model, ClinicalTextModel


@lru_cache(maxsize=1)
def get_available_image_backbones() -> tuple[str, ...]:
    """Registered CNN/ViT/Swin backbone names with a matching checkpoint on disk
    (excludes binary/multitask variants and the clinical/fusion checkpoints,
    which share the models/best_*.pt naming convention but aren't image backbones)."""
    found = set()
    for path in MODELS_DIR.glob("best_*.pt"):
        stem = path.stem.removeprefix("best_")
        if stem in AVAILABLE_MODELS:
            found.add(stem)
    return tuple(sorted(found))


@lru_cache(maxsize=1)
def get_image_model() -> KneeXRayClassifier | None:
    device = get_device()
    path = MODELS_DIR / "best_densenet121.pt"
    if not path.exists():
        return None
    return load_trained_model(path, device, num_classes=5)


@lru_cache(maxsize=1)
def get_clinical_model() -> ClinicalTextModel | None:
    device = get_device()
    path = MODELS_DIR / "best_clinical.pt"
    if not path.exists():
        return None
    return load_trained_clinical_model(path, device, num_classes=5)


@lru_cache(maxsize=1)
def get_fusion_model():
    device = get_device()
    path = MODELS_DIR / "best_fusion.pt"
    if not path.exists():
        return None
    from kneevision.fusion import load_trained_fusion_model
    return load_trained_fusion_model(path, device, num_classes=5)


@lru_cache(maxsize=1)
def get_rehab_recommender():
    from kneevision.rag import RehabRecommender
    return RehabRecommender.from_guidelines_dir(GUIDELINES_DIR)
