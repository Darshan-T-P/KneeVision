"""KneeVision++ demo — Streamlit app for multimodal knee OA diagnosis.

Run with:  uv run --extra dev streamlit run streamlit_app.py
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
bert_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--emilyalsentzer--Bio_ClinicalBERT"
if (Path(__file__).resolve().parent / "models" / "best_densenet121.pt").exists() and bert_cache.exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

# Bridge Streamlit secrets (local .streamlit/secrets.toml or Streamlit Cloud -> Secrets) into environment variables
# BEFORE kneevision.config.settings is imported, so XAI_API_KEY / LLM_* reach the Grok client. Env vars win if both are set.
try:
    for _key in ("LLM_PROVIDER", "XAI_API_KEY", "XAI_MODEL", "XAI_BASE_URL", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        if _key in st.secrets:
            os.environ.setdefault(_key, str(st.secrets[_key]))
except Exception:  # no secrets file configured: fall back to local Ollama / guideline excerpts
    pass

from kneevision.data.bilateral import (
    detect_bilateral_radiograph,
    detect_knee_prosthesis,
    extract_bilateral_knees,
    split_bilateral_halves,
)
from kneevision.data.joint_window import DEFAULT_JOINT_Y, knee_windows, multiscale_report
from kneevision.data.transforms import build_val_transform, val_transform
from kneevision.models.image_model import load_trained_model
from kneevision.training.losses import ordinal_to_probs
from kneevision.utils.helpers import get_device
from kneevision.xai import gradcam_explain, lime_explain, scorecam_explain

MODELS_DIR = Path(__file__).resolve().parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
DEMO_DIR = Path(__file__).resolve().parent / "data" / "raw" / "test"
DEMO_BILATERAL_PATH = Path(__file__).resolve().parent / "data" / "demo" / "bilateral_standing_ap.png"
DEMO_NATIVE_PATH = Path(__file__).resolve().parent / "data" / "demo" / "sample_native_crop.png"
DEMO_PROSTHESIS_PATH = Path(__file__).resolve().parent / "data" / "demo" / "sample_prosthetic_crop.png"
KL_LABELS = {0: "Normal", 1: "Doubtful", 2: "Mild", 3: "Moderate", 4: "Severe"}

CHECKPOINT_REPO = "Darshan13/KneeVision-models"
REQUIRED_CHECKPOINTS = [
    "best_densenet121.json", "best_densenet121.pt",
    "best_efficientnet-b4.json", "best_efficientnet-b4.pt",
    "best_convnext_small_binary.json", "best_convnext_small_binary.pt",
    "best_clinical.pt",
    "best_fusion.pt",
    # Final locked CORAL ordinal model (skipped gracefully if not yet on the HF repo)
    "best_densenet121_ordinal_ordinal_soft_mixup_a04.json", "best_densenet121_ordinal_ordinal_soft_mixup_a04.pt",
    # Best X-ray model (CORAL + OARSI auxiliary heads) and the domain-robust variant. The Swin-T and ConvNeXt-Tiny
    # checkpoints (~115 MB each) are used locally when present and are not fetched on the cloud.
    "best_densenet121_multitask_ordinal_soft_mixup_a04.json", "best_densenet121_multitask_ordinal_soft_mixup_a04.pt",
    "best_densenet121_ordinal_short40_robust_seed42.json", "best_densenet121_ordinal_short40_robust_seed42.pt",
]
# Checkpoint file -> dropdown label, best first (the first available one is the default).
IMAGE_MODEL_LABELS = {
    "best_densenet121_multitask_ordinal_soft_mixup_a04.pt": "DenseNet121 · CORAL + OARSI aux heads (best)",
    "best_densenet121_ordinal_ordinal_soft_mixup_a04.pt": "DenseNet121 · CORAL ordinal (locked)",
    "best_densenet121_ordinal_short40_robust_seed42.pt": "DenseNet121 · CORAL, domain-robust (40-epoch)",
    "best_swin_t_ordinal_ordinal_soft_mixup_a04.pt": "Swin-T · CORAL ordinal",
    "best_convnext_tiny_ordinal_ordinal_soft_mixup_a04.pt": "ConvNeXt-Tiny · CORAL ordinal",
    "best_densenet121_multitask.pt": "DenseNet121 · multitask, focal (older)",
    "best_densenet121_focal_nomixup.pt": "DenseNet121 · focal, no MixUp",
    "best_densenet121.pt": "DenseNet121 · standard 5-class",
    "best_efficientnet-b4.pt": "EfficientNet-B4 · standard 5-class",
}
STANDARD_DENSENET_LABEL = IMAGE_MODEL_LABELS["best_densenet121.pt"]
# Checkpoints trained with per-image standardization (--robust-aug) need the matching input transform.
ROBUST_INPUT_FILES = {"best_densenet121_ordinal_short40_robust_seed42.pt"}
# The default ensemble is the two DenseNet121 CORAL models (a fixed, simple choice; adding every loaded model would mix
# in older or differently-normalised models).
ENSEMBLE_LABELS = ["DenseNet121 · CORAL + OARSI aux heads (best)", "DenseNet121 · CORAL ordinal (locked)"]
ENSEMBLE_OPTION = "Ensemble (two CORAL DenseNet121 models)"


def model_tf(model):
    """Input transform matching how this model was trained."""
    return getattr(model, "val_tf", val_transform)


def ensemble_models(image_models: dict) -> list:
    models = [image_models[label] for label in ENSEMBLE_LABELS if label in image_models]
    return models or list(image_models.values())[:1]


def load_app_results():
    """Verified numbers shown in the app, generated by scripts/build_app_results.py (tracked in docs/)."""
    try:
        return json.loads((Path(__file__).resolve().parent / "docs" / "app_results.json").read_text())
    except (OSError, ValueError):
        return None


def ensure_checkpoints() -> None:
    """Pull inference checkpoints from the companion HF Hub model repo on first run —
    they're too large for git and aren't bundled with the app (e.g. on Spaces)."""
    missing = [f for f in REQUIRED_CHECKPOINTS if not (MODELS_DIR / f).exists()]
    if not missing:
        return
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

    MODELS_DIR.mkdir(exist_ok=True)
    for filename in missing:
        try:
            hf_hub_download(repo_id=CHECKPOINT_REPO, filename=filename, local_dir=str(MODELS_DIR))
        except (EntryNotFoundError, HfHubHTTPError, OSError) as exc:
            # Runs before st.set_page_config() — an st.* call here would crash the whole app
            # (Streamlit requires set_page_config to be the first command). The individual
            # loaders below already degrade gracefully when a checkpoint is missing.
            print(f"[ensure_checkpoints] could not fetch {filename} from {CHECKPOINT_REPO}: {exc}")


ensure_checkpoints()

DEMO_REPORTS = {
    0: ("FINDINGS: The medial and lateral femorotibial joint spaces are well preserved. "
        "No osteophyte formation, subchondral sclerosis, or cystic change is identified. "
        "Patellofemoral compartment is normal. Periarticular soft tissues are unremarkable.\n\n"
        "IMPRESSION: No significant radiographic abnormality."),
    1: ("FINDINGS: Minute marginal osteophyte formation is noted at the medial femoral condyle. "
        "Joint spaces remain preserved. No definite subchondral sclerosis or cyst formation.\n\n"
        "IMPRESSION: Minimal marginal osteophyte formation; joint spaces are maintained."),
    2: ("FINDINGS: Small-to-moderate marginal osteophytes arise from the medial tibial plateau and "
        "femoral condyle. Subtle narrowing of the medial compartment is present. Possible early "
        "subchondral sclerosis. Intercondylar eminence is sharp.\n\n"
        "IMPRESSION: Mild medial compartment degenerative change."),
    3: ("EXAM: Three-view knee radiographs.\n\n"
        "COMPARISON: No prior study is available.\n\n"
        "FINDINGS: There is mild varus alignment. Definite medial femorotibial joint-space narrowing "
        "is present with several moderate marginal osteophytes arising from the medial femoral condyle "
        "and tibial plateau. Moderate subchondral sclerosis and small subchondral cystic areas with "
        "well-defined sclerotic margins are seen in the medial compartment. The lateral femorotibial "
        "joint space is relatively maintained. Small patellofemoral marginal osteophytes are present. "
        "No acute fracture or dislocation is identified. No sizable joint effusion is seen.\n\n"
        "IMPRESSION: Moderate osteoarthritic change, greatest in the medial femorotibial compartment, "
        "with mild varus alignment."),
    4: ("FINDINGS: Severe near-complete loss of the medial joint space with a bone-on-bone appearance. "
        "Large osteophytes project from all compartments. Marked subchondral sclerosis with defined "
        "pseudocysts. Gross deformity of the femoral and tibial articular surfaces.\n\n"
        "IMPRESSION: Advanced tricompartmental degenerative change with articular deformity."),
}

GRADE_FEATURES = {
    0: "Normal joint — intact joint space, no osteophytes or sclerosis.",
    1: "Doubtful — minute osteophytes, joint space still preserved.",
    2: "Mild — small osteophytes with possible joint space narrowing.",
    3: "Moderate — definite narrowing, moderate osteophytes, sclerosis.",
    4: "Severe — bone-on-bone, large osteophytes, marked deformity.",
}

CLINICAL_EXAMPLES = {
    "Custom report": "",
    "Sample report A": DEMO_REPORTS[0],
    "Sample report B": DEMO_REPORTS[1],
    "Sample report C": DEMO_REPORTS[2],
    "Sample report D": DEMO_REPORTS[3],
    "Sample report E": DEMO_REPORTS[4],
}
CLINICAL_EXAMPLE_REFERENCE = {
    "Sample report A": 0,
    "Sample report B": 1,
    "Sample report C": 2,
    "Sample report D": 3,
    "Sample report E": 4,
}

REHAB_EXAMPLES = {
    "Custom patient context": "",
    "Active adult · intermittent pain on stairs": "58-year-old, BMI 27, intermittent knee pain on stairs; remains active and reports no instability.",
    "Persistent symptoms · reduced walking tolerance": "71-year-old, BMI 32, persistent weight-bearing pain and difficulty walking more than 10 minutes; reports occasional instability.",
    "Low symptoms · morning stiffness": "53-year-old, BMI 25, mild morning stiffness lasting about 15 minutes; no swelling and normal daily walking.",
}

REHAB_PRIMARY_SOURCES = [
    ("OARSI 2019 non-surgical OA management guidelines", "https://oarsi.org/education/oarsi-guidelines"),
    ("AAOS 2021 Management of Osteoarthritis of the Knee guideline", "https://www.aaos.org/globalassets/quality-and-practice-resources/osteoarthritis-of-the-knee/oak3cpg.pdf"),
    ("ACR / Arthritis Foundation 2019 OA management guideline", "https://pubmed.ncbi.nlm.nih.gov/?term=2019+ACR+Arthritis+Foundation+Guideline+Osteoarthritis+Knee"),
    ("CDC arthritis self-management guidance", "https://www.cdc.gov/arthritis/hcp/self-management/index.html"),
    ("Arthritis Foundation OA treatment guideline overview", "https://www.arthritis.org/diseases/more-about/guidelines-for-osteoarthritis-treatments"),
]


def _load_selected_example(select_key: str, input_key: str, examples: dict[str, str]) -> None:
    selected = st.session_state.get(select_key, "")
    st.session_state[input_key] = examples.get(selected, "")


def show_rehab_sources(result) -> None:
    with st.expander(f"Sources and retrieved evidence ({len(result.retrieved_chunks)} excerpts)", expanded=False):
        st.markdown(
            "The app retrieves from original summaries in `data/guidelines/`, which synthesize the "
            "public references below. These summaries are not verbatim guideline text and are not "
            "individually endorsed by each source."
        )
        st.markdown("**Primary references**")
        for title, url in REHAB_PRIMARY_SOURCES:
            st.markdown(f"- [{title}]({url})")

        st.markdown("**Retrieved project summaries used for this response**")
        if not result.retrieved_chunks:
            st.info("No matching guideline excerpts were retrieved.")
        for chunk in result.retrieved_chunks:
            st.markdown(f"**{chunk.heading}** · `{chunk.source_path}`")
            st.info(chunk.text)


@st.cache_data(show_spinner="Indexing demo X-rays...")
def demo_images() -> dict[int, str]:
    out: dict[int, str] = {}
    if not DEMO_DIR.exists():
        return out
    for grade in range(5):
        folder = DEMO_DIR / str(grade)
        files = sorted(folder.glob("*.png")) or sorted(folder.glob("*.[jp][pn]g"))
        if files:
            # deterministic pick: hash-free middle file keeps demos stable across restarts
            out[grade] = str(files[len(files) // 2])
    return out

# ----------------------------------------------------------------------------
# Design system
# ----------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

section.main > div {
    padding-top: 1.2rem;
    max-width: 1300px;
}

/* Hero Section */
.kvp-hero {
    background: linear-gradient(135deg, #091e36 0%, #0c2a47 45%, #08172c 100%);
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    color: #ffffff;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 36px -8px rgba(0, 0, 0, 0.45), 0 0 30px -10px rgba(14, 165, 233, 0.2);
    position: relative;
    overflow: hidden;
}
.kvp-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #38bdf8 25%, #2dd4bf 75%, transparent);
}
.kvp-hero-title-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-bottom: 0.4rem;
}
.kvp-hero h1 {
    color: #ffffff !important;
    margin: 0 !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em !important;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.kvp-status-pill {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #6ee7b7;
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.kvp-hero p {
    color: #94a3b8 !important;
    margin: 0 !important;
    font-size: 1.02rem !important;
    line-height: 1.6 !important;
    max-width: 860px;
}
.kvp-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1.25rem;
}
.kvp-badges span {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(56, 189, 248, 0.22);
    border-radius: 999px;
    padding: 0.32rem 0.85rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: #bae6fd;
    backdrop-filter: blur(8px);
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
.kvp-badges span:hover {
    background: rgba(14, 165, 233, 0.18);
    border-color: rgba(56, 189, 248, 0.5);
    color: #ffffff;
    transform: translateY(-1px);
}

/* Glassmorphic Elevated Cards */
.kvp-card {
    background: rgba(17, 24, 39, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 16px;
    padding: 1.35rem 1.5rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35), 0 0 0 1px rgba(255, 255, 255, 0.03) inset;
    height: 100%;
    color: #e2e8f0;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.kvp-card:hover {
    border-color: rgba(56, 189, 248, 0.32);
    box-shadow: 0 12px 28px -4px rgba(14, 165, 233, 0.18), 0 0 0 1px rgba(56, 189, 248, 0.1) inset;
    transform: translateY(-2px);
}
.kvp-card h4 {
    margin: 0 0 0.75rem 0 !important;
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.kvp-grade {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.03em;
}
.kvp-sub {
    color: #94a3b8;
    font-size: 0.88rem;
    margin-top: 0.25rem;
}
.kvp-bar {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    height: 0.65rem;
    overflow: hidden;
    margin-top: 0.75rem;
    border: 1px solid rgba(255, 255, 255, 0.04);
}
.kvp-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #0ea5e9, #10b981);
    transition: width 0.4s ease-out;
}

/* Metric Stat Box */
.kvp-stat-container {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: calc(100% - 1.8rem);
}
.kvp-stat-number {
    font-size: 2.2rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0.2rem 0 0.4rem 0;
}
.kvp-stat-number span.unit {
    font-size: 1.05rem;
    font-weight: 600;
    color: #38bdf8;
    margin-left: 0.3rem;
}
.kvp-stat-desc {
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.45;
}
.kvp-pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: #38bdf8;
    border-radius: 6px;
    padding: 0.2rem 0.55rem;
    font-size: 0.72rem;
    font-weight: 700;
    margin-top: 0.7rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Multimodal Pipeline Architecture Box */
.kvp-pipeline-card {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin: 1.5rem 0;
    box-shadow: 0 8px 30px -4px rgba(0, 0, 0, 0.3);
}
.kvp-pipeline-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1.25rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.kvp-pipeline-header h4 {
    margin: 0 !important;
    font-size: 0.85rem !important;
    font-weight: 800 !important;
    color: #e2e8f0 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.kvp-flow-grid {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
}
.flow-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
}
.flow-node {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(56, 189, 248, 0.28);
    color: #f1f5f9;
    border-radius: 12px;
    padding: 0.7rem 1.15rem;
    font-size: 0.88rem;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    min-width: 145px;
    transition: all 0.2s ease;
}
.flow-node:hover {
    border-color: rgba(56, 189, 248, 0.6);
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(14, 165, 233, 0.2);
}
.flow-node small {
    font-weight: 500;
    color: #38bdf8;
    font-size: 0.73rem;
    letter-spacing: 0.02em;
}
.flow-node.fusion {
    background: linear-gradient(135deg, rgba(88, 28, 135, 0.35) 0%, rgba(59, 130, 246, 0.25) 100%);
    border-color: rgba(168, 85, 247, 0.5);
    color: #faf5ff;
}
.flow-node.fusion small {
    color: #c084fc;
}
.flow-node.output {
    background: linear-gradient(135deg, rgba(161, 98, 7, 0.3) 0%, rgba(202, 138, 4, 0.2) 100%);
    border-color: rgba(251, 191, 36, 0.5);
    color: #fef9c3;
}
.flow-node.output small {
    color: #fde047;
}
.flow-arrow {
    color: #64748b;
    font-size: 1.15rem;
    font-weight: 800;
}
.flow-spacer {
    flex: 0 0 3.2rem;
}

/* Feature Grid Cards */
.kvp-features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1rem;
    margin: 1rem 0 1.8rem 0;
}
.kvp-feat-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.15rem 1.25rem;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
.kvp-feat-card:hover {
    border-color: rgba(56, 189, 248, 0.35);
    background: rgba(15, 23, 42, 0.85);
    transform: translateY(-2px);
}
.kvp-feat-icon {
    font-size: 1.4rem;
    margin-bottom: 0.5rem;
}
.kvp-feat-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.35rem;
}
.kvp-feat-text {
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.45;
}

/* Styled HTML Results Table */
.kvp-table-wrap {
    overflow-x: auto;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(15, 23, 42, 0.7);
    margin: 0.8rem 0 0.5rem 0;
}
.kvp-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    text-align: left;
}
.kvp-table th {
    background: rgba(15, 23, 42, 0.95);
    color: #94a3b8;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 0.74rem;
    letter-spacing: 0.08em;
    padding: 0.85rem 1.1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}
.kvp-table td {
    padding: 0.9rem 1.1rem;
    color: #e2e8f0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.kvp-table tr:last-child td {
    border-bottom: none;
}
.kvp-table tr:hover td {
    background: rgba(255, 255, 255, 0.02);
}
.kvp-tag-pill {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 600;
}
.kvp-tag-blue { background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.25); }
.kvp-tag-green { background: rgba(52, 211, 153, 0.12); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.25); }
.kvp-tag-purple { background: rgba(168, 85, 247, 0.12); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.25); }

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2eaf0 !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    background: transparent;
    border: 1px solid transparent;
    padding: 0.55rem 0.85rem;
    border-radius: 10px;
    margin-bottom: 0.3rem;
    transition: all 0.2s ease;
    cursor: pointer;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: #f2f8fa;
    border-color: #dce7ee;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
    background: #eaf6f8 !important;
    border: 1px solid #9cced6 !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) p {
    color: #087e8b !important;
    font-weight: 700 !important;
}
.kvp-sidebar-header {
    padding: 0.2rem 0.1rem 1rem 0.1rem;
    border-bottom: 1px solid #e2eaf0;
    margin-bottom: 0.9rem;
}
.kvp-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.kvp-logo-badge {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #0ea5e9, #10b981);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    box-shadow: 0 4px 14px rgba(14, 165, 233, 0.4);
}
.kvp-brand-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.025em;
    line-height: 1.1;
}
.kvp-brand-tag {
    font-size: 0.65rem;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}
.kvp-brand-desc {
    color: #94a3b8;
    font-size: 0.8rem;
    margin-top: 0.7rem;
    line-height: 1.45;
}
.kvp-brand-desc span {
    color: #64748b;
    font-size: 0.75rem;
}
.kvp-sidebar-footer {
    margin-top: 1.2rem;
    padding: 0.85rem;
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.06);
}
.kvp-footer-status {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #34d399;
    margin-bottom: 0.4rem;
}
.dot-online {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #10b981;
    box-shadow: 0 0 8px #10b981;
    display: inline-block;
}
.kvp-footer-meta {
    font-size: 0.72rem;
    color: #64748b;
    line-height: 1.4;
}
.kvp-footer-meta code {
    background: rgba(255, 255, 255, 0.06);
    color: #94a3b8;
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
    font-size: 0.68rem;
}
.chip-ok {
    background: rgba(16, 185, 129, 0.15) !important;
    color: #34d399 !important;
    border: 1px solid rgba(16, 185, 129, 0.35) !important;
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-weight: 700;
    display: inline-block;
    font-size: 0.78rem;
}
.chip-warn {
    background: rgba(245, 158, 11, 0.15) !important;
    color: #fbbf24 !important;
    border: 1px solid rgba(245, 158, 11, 0.35) !important;
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-weight: 700;
    display: inline-block;
    font-size: 0.78rem;
}

/* White theme */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background: #ffffff !important;
    color: #172b3a !important;
}
[data-testid="stHeader"] {
    border-bottom: 1px solid #e5edf2;
}
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2eaf0 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #172b3a !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #526779 !important;
}
.kvp-hero,
.kvp-card,
.kvp-pipeline-card,
.flow-node,
.kvp-feat-card,
.kvp-table-wrap,
.kvp-sidebar-footer {
    background: #ffffff !important;
    color: #172b3a !important;
    border-color: #dce7ee !important;
    box-shadow: 0 5px 18px rgba(34, 65, 83, 0.08) !important;
}
.kvp-hero {
    background: linear-gradient(120deg, #ffffff 0%, #f2fbfc 100%) !important;
}
.kvp-hero h1,
.kvp-hero p,
.kvp-card,
.kvp-card h4,
.kvp-sub,
.kvp-stat-number,
.kvp-stat-desc,
.kvp-pipeline-header h4,
.flow-node,
.kvp-feat-title,
.kvp-feat-text,
.kvp-table td,
.kvp-brand-title,
.kvp-brand-desc,
.kvp-footer-meta {
    color: #172b3a !important;
}
.kvp-hero p,
.kvp-sub,
.kvp-stat-desc,
.kvp-feat-text,
.kvp-brand-desc,
.kvp-footer-meta {
    color: #526779 !important;
}
.kvp-status-pill,
.kvp-footer-status,
.kvp-brand-tag,
.kvp-grade,
.kvp-stat-number span.unit,
.kvp-pill-badge {
    color: #087e8b !important;
}
.kvp-status-pill {
    background: #e8f7f4 !important;
    border-color: #a9ddd1 !important;
}
.kvp-tag-blue {
    color: #075985 !important;
}
.kvp-tag-green {
    color: #166534 !important;
}
.kvp-tag-purple {
    color: #6b3fa0 !important;
}
.chip-ok {
    color: #166534 !important;
}
.chip-warn {
    color: #92400e !important;
}
.kvp-badges span,
.kvp-footer-meta code,
.kvp-table th {
    background: #f3f8fa !important;
    color: #345367 !important;
    border-color: #dce7ee !important;
}
.kvp-pipeline-header {
    border-bottom-color: #e2eaf0 !important;
}
.flow-node.fusion {
    background: #eff8fb !important;
    border-color: #b9dce8 !important;
    color: #164e63 !important;
}
.flow-node.fusion small {
    color: #087e8b !important;
}
.flow-node.output {
    background: #f5faf5 !important;
    border-color: #c8dfcb !important;
    color: #315c3a !important;
}
.flow-node.output small {
    color: #3d7a4b !important;
}
.kvp-table td {
    border-bottom-color: #e7eef2 !important;
}
.kvp-table tr:hover td,
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: #f2f8fa !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
    background: #eaf6f8 !important;
    border-color: #9cced6 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) p {
    color: #087e8b !important;
}
input,
textarea,
[data-baseweb="input"],
[data-baseweb="select"] > div,
[data-testid="stNumberInput"] button,
[data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff !important;
    color: #172b3a !important;
    border-color: #cbd9e1 !important;
}
[data-testid="stMarkdownContainer"] [style*="background:rgba(15,23,42"],
[data-testid="stMarkdownContainer"] [style*="background:#1e293b"],
[data-testid="stMarkdownContainer"] [style*="background:linear-gradient(120deg,#064e3b"] {
    background: #ffffff !important;
    color: #172b3a !important;
    border-color: #dce7ee !important;
}
[data-testid="stMarkdownContainer"] [style*="color:#f8fafc"],
[data-testid="stMarkdownContainer"] [style*="color:#f1f5f9"],
[data-testid="stMarkdownContainer"] [style*="color:#e2e8f0"],
[data-testid="stMarkdownContainer"] [style*="color:#ffffff"],
[data-testid="stMarkdownContainer"] [style*="color:#94a3b8"] {
    color: #172b3a !important;
}

/* ---- Buttons (every page): readable on the light layout ---- */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondaryFormSubmit"],
[data-testid="stFileUploader"] button {
    background: #ffffff !important;
    color: #087e8b !important;
    border: 1px solid #9cced6 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transition: background .15s, border-color .15s, color .15s;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover,
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stFileUploader"] button:hover {
    background: #eaf6f8 !important;
    border-color: #087e8b !important;
    color: #066a75 !important;
}
.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible,
[data-testid="stFileUploader"] button:focus-visible {
    box-shadow: 0 0 0 3px rgba(8, 126, 139, .28) !important;
    outline: none !important;
}
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"],
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"] {
    background: #087e8b !important;
    color: #ffffff !important;
    border: 1px solid #087e8b !important;
}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stBaseButton-primaryFormSubmit"]:hover {
    background: #066a75 !important;
    border-color: #066a75 !important;
    color: #ffffff !important;
}
.stButton > button:disabled,
.stDownloadButton > button:disabled,
[data-testid="stBaseButton-secondary"]:disabled,
[data-testid="stBaseButton-primary"]:disabled {
    background: #eef3f5 !important;
    color: #8ba0a8 !important;
    border-color: #d9e4e8 !important;
}
.stButton > button p,
.stDownloadButton > button p,
.stFormSubmitButton > button p,
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button span {
    color: inherit !important;
}
</style>
"""


def hero():
    st.markdown(
        """
        <div class="kvp-hero">
            <div class="kvp-hero-title-row">
                <h1>🦵 KneeVision++ <span style="font-weight:400;color:#94a3b8;font-size:1.4rem;">· Multimodal OA Diagnosis</span></h1>
                <div class="kvp-status-pill"><span class="dot-online"></span> Research Models Active</div>
            </div>
            <p>Research prototype for knee osteoarthritis: Kellgren–Lawrence (KL 0–4) grading from radiographs, screening from report text generated from structured OAI findings, and late-fusion decision support. Not a diagnostic device.</p>
            <div class="kvp-badges">
                <span>🩻 KL Grades 0–4</span>
                <span>⚡ CORAL DenseNet121 Models</span>
                <span>🧬 BioClinicalBERT</span>
                <span>🔍 Grad-CAM &middot; Score-CAM &middot; LIME</span>
                <span>📊 MLflow Monitored</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body_html: str):
    return f'<div class="kvp-card"><h4>{title}</h4>{body_html}</div>'


def metric_card(title: str, value: str, unit: str, subtitle: str, badge_text: str, icon: str) -> str:
    return f"""
    <div class="kvp-card">
        <h4><span>{icon}</span> {title}</h4>
        <div class="kvp-stat-container">
            <div>
                <div class="kvp-stat-number">{value}<span class="unit">{unit}</span></div>
                <div class="kvp-stat-desc">{subtitle}</div>
            </div>
            <div>
                <span class="kvp-pill-badge">{badge_text}</span>
            </div>
        </div>
    </div>
    """


def grade_card(title: str, pred: int | None, conf: float | None, accent: str) -> str:
    if pred is None:
        return card(title, '<div class="kvp-sub">—</div>')
    pct = f"{conf:.1%}" if conf is not None else "—"
    fill_w = int((conf or 0) * 100)
    return card(
        title,
        f'<div class="kvp-grade" style="color:{accent}">KL {pred}</div>'
        f'<div class="kvp-sub" style="font-size:1.05rem;font-weight:700;color:#f8fafc;margin-bottom:0.35rem;">{KL_LABELS[pred]}</div>'
        f'<div class="kvp-bar"><div class="kvp-fill" style="width:{fill_w}%;background:linear-gradient(90deg, {accent}, #38bdf8)"></div></div>'
        f'<div class="kvp-sub" style="margin-top:0.45rem;display:flex;justify-content:space-between;"><span>Confidence</span><b style="color:#f8fafc">{pct}</b></div>',
    )


def prosthesis_card(title: str) -> str:
    body = (
        '<div class="kvp-grade" style="color:#f59e0b;font-size:2.0rem;letter-spacing:-0.5px;">🦾 TKA Implant</div>'
        '<div class="kvp-sub" style="font-size:1.05rem;font-weight:700;color:#f8fafc;margin-bottom:0.4rem;">Total Knee Arthroplasty (Replaced)</div>'
        '<div style="margin-bottom:0.75rem;"><span class="chip-warn">Artificial Prosthesis</span></div>'
        '<div class="kvp-sub" style="color:#94a3b8;font-size:0.86rem;line-height:1.45;">'
        '<b>Clinical Protocol Note:</b> Kellgren-Lawrence (KL 0–4) grading applies strictly to native biological joints. '
        'Femoral metallic shield, tibial tray, and polyethylene spacer hardware replace biological cartilage.'
        '</div>'
    )
    return card(title, body)


def fusion_flow_diagram():
    st.markdown(
        """
        <div class="kvp-pipeline-card">
          <div class="kvp-pipeline-header">
            <h4>🔀 Multimodal Late-Fusion Pipeline Architecture</h4>
            <span class="kvp-tag-pill kvp-tag-purple">Decision Fusion α = 0.50</span>
          </div>
          <div class="kvp-flow-grid">
            <div class="flow-row">
              <div class="flow-node">
                🩻 Knee X-ray
                <small>224×224 normalized radiograph</small>
              </div>
              <div class="flow-arrow">&rarr;</div>
              <div class="flow-node">
                X-ray Ensemble
                <small>DenseNet121 &middot; EfficientNet-B4</small>
              </div>
              <div class="flow-arrow">&rarr;</div>
              <div class="flow-node">
                P(KL 0&ndash;4)
                <small>imaging modality softmax</small>
              </div>
            </div>
            <div class="flow-row">
              <div class="flow-node">
                📝 Clinical Report
                <small>report text built from structured OAI findings</small>
              </div>
              <div class="flow-arrow">&rarr;</div>
              <div class="flow-node">
                BioClinicalBERT
                <small>frozen encoder + classification head</small>
              </div>
              <div class="flow-arrow">&rarr;</div>
              <div class="flow-node">
                P(KL 0&ndash;4)
                <small>clinical NLP modality softmax</small>
              </div>
            </div>
            <div class="flow-row">
              <div class="flow-spacer"></div>
              <div class="flow-arrow">&#8618;</div>
              <div class="flow-node fusion">
                &#8853; Late Fusion Layer
                <small>P<sub>fused</sub> = (1&minus;&alpha;)&middot;P<sub>img</sub> + &alpha;&middot;P<sub>text</sub></small>
              </div>
              <div class="flow-arrow">&rarr;</div>
              <div class="flow-node output">
                🎯 Final KL Grade
                <small>argmax + calibrated confidence</small>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Data access helpers
# ----------------------------------------------------------------------------


def checkpoint_metas() -> list[dict]:
    metas = []
    for path in sorted(MODELS_DIR.glob("best_*.json")):
        if ".cm." in path.name or ".test." in path.name:
            continue
        try:
            meta = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict) or "model_name" not in meta:
            continue
        pt_file = path.stem + ".pt"
        if not (MODELS_DIR / pt_file).exists():
            continue
        meta["file"] = pt_file
        metas.append(meta)
    return metas


def report_text(name: str) -> str:
    path = REPORTS_DIR / name / "classification_report.txt"
    return path.read_text().strip() if path.exists() else "No report found — run scripts/evaluate.py first."


def report_image(name: str) -> str | None:
    path = REPORTS_DIR / name / "confusion_matrix.png"
    return str(path) if path.exists() else None


def report_accuracy(name: str) -> float | None:
    match = re.search(r"accuracy\s+([\d.]+)", report_text(name))
    return float(match.group(1)) if match else None


@st.cache_data(ttl=60, show_spinner="Querying MLflow...")
def mlflow_runs_table() -> pd.DataFrame:
    from mlflow.tracking import MlflowClient

    db = Path(__file__).resolve().parent / "mlflow.db"
    client = MlflowClient(tracking_uri=f"sqlite:///{db}")
    rows = []
    for exp in client.search_experiments():
        for run in client.search_runs([exp.experiment_id], order_by=["start_time DESC"]):
            params, metrics = run.data.params, run.data.metrics
            rows.append({
                "Run": run.info.run_name or run.info.run_id[:8],
                "Model": params.get("model_name") or params.get("models", "—"),
                "Binary": params.get("binary", ""),
                "Best val κ": round(metrics["best_kappa"], 4) if "best_kappa" in metrics else None,
                "Test acc": round(metrics["accuracy"], 4) if "accuracy" in metrics else None,
                "ROC-AUC": round(metrics["roc_auc"], 4) if "roc_auc" in metrics else None,
                "Epochs": params.get("num_epochs") or params.get("epochs", ""),
                "Time (min)": round(metrics["total_time_sec"] / 60, 1) if "total_time_sec" in metrics else None,
                "Started": pd.to_datetime(run.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M"),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Model loading + inference
# ----------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading image models...")
def load_image_models():
    device = get_device()
    available = {}
    # Explicit, uniquely-labelled list. Keying by backbone name made checkpoints that share a
    # backbone overwrite each other, so a label could silently load a different model locally
    # than on the cloud. Missing files are skipped, so the cloud (fewer checkpoints) still works.
    for filename, label in IMAGE_MODEL_LABELS.items():
        path = MODELS_DIR / filename
        if not path.exists():
            continue
        try:
            model = load_trained_model(path, device, num_classes=5)
            if filename in ROBUST_INPUT_FILES:
                model.val_tf = build_val_transform(robust=True)
            available[label] = model
        except (OSError, RuntimeError, ValueError, KeyError, ImportError) as exc:
            print(f"[load_image_models] Could not load {path}: {exc}")
    return device, available


@st.cache_resource(show_spinner="Loading binary screening model...")
def load_binary_model():
    device = get_device()
    meta = next((m for m in checkpoint_metas()
                 if m.get("binary") and m["model_name"] == "convnext_small"), None)
    meta = meta or next((m for m in checkpoint_metas() if m.get("binary")), None)
    if meta is None:
        return None, None
    pt_path = MODELS_DIR / meta["file"]
    if not pt_path.exists():
        return None, None
    try:
        model = load_trained_model(pt_path, device, num_classes=2)
        return model, meta["model_name"]
    except (OSError, RuntimeError, ValueError, KeyError, ImportError) as exc:
        print(f"[load_binary_model] Could not load {pt_path}: {exc}")
        return None, None


@st.cache_resource(show_spinner="Loading clinical model...")
def load_clinical_model():
    device = get_device()
    from kneevision.clinical.model import ClinicalTextModel, load_trained_clinical_model

    trained = MODELS_DIR / "best_clinical.pt"
    try:
        if trained.exists():
            return load_trained_clinical_model(trained, device, num_classes=5), True
        model = ClinicalTextModel(num_classes=5, ordinal=False).to(device)
        return model, False
    except (OSError, RuntimeError, ValueError, KeyError, ImportError) as exc:
        print(f"[load_clinical_model] Error: {exc}")
        return None, False


@st.cache_resource(show_spinner="Loading multimodal fusion model...")
def load_fusion_model():
    device = get_device()
    from kneevision.fusion import load_trained_fusion_model

    trained = MODELS_DIR / "best_fusion.pt"
    try:
        if trained.exists():
            model = load_trained_fusion_model(trained, device, num_classes=5)
            return model, True
        return None, False
    except (OSError, RuntimeError, ValueError, KeyError, ImportError) as exc:
        print(f"[load_fusion_model] Could not load {trained}: {exc}")
        return None, False


@st.cache_resource(show_spinner="Loading rehab guideline index...")
def load_rehab_recommender():
    from kneevision.config.settings import GUIDELINES_DIR
    from kneevision.rag import RehabRecommender

    return RehabRecommender.from_guidelines_dir(GUIDELINES_DIR)


@torch.inference_mode()
def predict_image(model, image: Image.Image, device) -> tuple[int, float, np.ndarray]:
    model.eval()
    x = model_tf(model)(image).unsqueeze(0).to(device)
    logits = model(x)
    if getattr(model, "ordinal", False):
        probs = ordinal_to_probs(logits).squeeze(0)
    else:
        probs = torch.softmax(logits, dim=1).squeeze(0)
    pred = int(probs.argmax().item())
    return pred, float(probs[pred].item()), probs.cpu().numpy()


@torch.inference_mode()
def image_probs(image: Image.Image, models: list, device) -> np.ndarray:
    total = np.zeros(5)
    for model in models:
        _, _, p = predict_image(model, image, device)
        total += p
    return total / len(models)


@torch.inference_mode()
def clinical_probs(model, text: str, device) -> np.ndarray:
    enc = model._encode([text], device)
    logits = model(enc["input_ids"], enc["attention_mask"])
    if getattr(model, "ordinal", False):
        from kneevision.training.losses import ordinal_to_probs
        return ordinal_to_probs(logits).squeeze(0).cpu().numpy()
    return torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()


@torch.inference_mode()
def binary_oa_prob(model, image: Image.Image, device) -> float:
    model.eval()
    x = val_transform(image).unsqueeze(0).to(device)
    return float(torch.softmax(model(x), dim=1).squeeze(0)[1].item())


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------


def page_overview():
    hero()
    col_l, col_m, col_r = st.columns(3)
    with col_l:
        st.markdown(
            metric_card(
                title="Dataset Cohort",
                value="8,260",
                unit="X-rays",
                subtitle="OAI-derived Kaggle benchmark · Patient-disjoint train/val/test splits",
                badge_text="Standardized Split",
                icon="🩻",
            ),
            unsafe_allow_html=True,
        )
    with col_m:
        st.markdown(
            metric_card(
                title="Clinical Corpus",
                value="16,592",
                unit="reports",
                subtitle="Report text generated from structured OAI fields (radiographic findings, symptoms, demographics)",
                badge_text="BioClinicalBERT",
                icon="📝",
            ),
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown(
            metric_card(
                title="Best Screening",
                value="87.0%",
                unit="acc",
                subtitle="ConvNeXt-S binary OA triage detector (KL 0–1 vs 2–4) · AUC 0.946",
                badge_text="AUC 0.946",
                icon="🎯",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("### Try the demo")
    st.markdown(
        "1. **Guided Demo**: run the same reference X-ray through the image, clinical-text, and screening models. "
        "The report is synthetic and is not paired to the image.\n"
        "2. **Clinical Text**: choose one of five label-free report examples, then select **Predict KL grade**. "
        "The example category is a reference, not a promised prediction.\n"
        "3. **Multimodal Fusion**: select an example case to load its report and available reference X-ray, "
        "or upload/paste your own inputs.\n"
        "4. **Rehab Recommendation**: choose a KL grade and optional symptom profile, then generate guideline-based guidance."
    )
    st.caption("Use de-identified test examples only. This research demo is not for diagnosis or treatment decisions.")

    fusion_flow_diagram()

    st.markdown("### 🛠️ Architecture & Research Methodology")
    st.markdown(
        """
        <div class="kvp-features-grid">
            <div class="kvp-feat-card">
                <div class="kvp-feat-icon">🩻</div>
                <div class="kvp-feat-title">1. Data Architecture</div>
                <div class="kvp-feat-text">8,260 standardized radiographs across 5 KL severity grades plus 16.6k OAI clinical records rendered as report text (structured findings, not free-text radiology notes).</div>
            </div>
            <div class="kvp-feat-card">
                <div class="kvp-feat-icon">⚙️</div>
                <div class="kvp-feat-title">2. 5-Class KL Grading</div>
                <div class="kvp-feat-text">DenseNet121 with a cumulative ordinal (CORAL-style) head, soft targets under MixUp, optional OARSI auxiliary heads, and early stopping on validation Quadratic Weighted κ.</div>
            </div>
            <div class="kvp-feat-card">
                <div class="kvp-feat-icon">⚡</div>
                <div class="kvp-feat-title">3. Dedicated OA Screening</div>
                <div class="kvp-feat-text">High-throughput binary OA triage (KL 0-1 vs 2-4) powered by ConvNeXt-S achieving 87.0% accuracy and 0.946 ROC-AUC.</div>
            </div>
            <div class="kvp-feat-card">
                <div class="kvp-feat-icon">📊</div>
                <div class="kvp-feat-title">4. Soft Probability Grouping</div>
                <div class="kvp-feat-text">Clinical risk-tier aggregation of the 5-grade X-ray output into 2 groups (86.0% accuracy) and 3 groups (79.2%).</div>
            </div>
            <div class="kvp-feat-card">
                <div class="kvp-feat-icon">🎯</div>
                <div class="kvp-feat-title">5. Ensembling & Calibration</div>
                <div class="kvp-feat-text">Validation-tuned multi-model ensembling and test-time augmentation (TTA) optimizing ROC-AUC and Cohen's κ.</div>
            </div>
            <div class="kvp-feat-card">
                <div class="kvp-feat-icon">🔥</div>
                <div class="kvp-feat-title">6. Interpretable XAI Heatmaps</div>
                <div class="kvp-feat-text">Grad-CAM, Score-CAM and LIME attention maps with measured insertion/deletion faithfulness (indicative, not diagnostic).</div>
            </div>
            <div class="kvp-feat-card">
                <div class="kvp-feat-icon">🔀</div>
                <div class="kvp-feat-title">7. Multimodal Late Fusion</div>
                <div class="kvp-feat-text">Real-time fusion of radiograph visual features and BioClinicalBERT clinical note representations.</div>
            </div>
            <div class="kvp-feat-card">
                <div class="kvp-feat-icon">📈</div>
                <div class="kvp-feat-title">8. MLflow Lifecycle Tracking</div>
                <div class="kvp-feat-text">End-to-end experiment logging tracking parameters, checkpoint weights, confusion matrices, and validation metrics.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🏆 Headline Test-Set Benchmark Results")
    _ar = load_app_results()
    if _ar:
        _xb = _ar["branch_tasks"]["xray_final"]
        acc_5c_str, acc_gb_str, acc_g3_str = f"{_xb['five_class'][0]:.1f}%", f"{_xb['binary'][0]:.1f}%", f"{_xb['three_class'][0]:.1f}%"
        qwk_str, auc_gb_str = f"{_xb['qwk'][0]:.3f} (test)", f"{_xb['binary_auc'][0]:.3f}"
    else:
        acc_5c_str, acc_gb_str, acc_g3_str, qwk_str, auc_gb_str = "—", "—", "—", "—", "—"

    table_html = f"""
    <div class="kvp-table-wrap">
        <table class="kvp-table">
            <thead>
                <tr>
                    <th>Clinical Task</th>
                    <th>Model Architecture</th>
                    <th>Test Accuracy</th>
                    <th>ROC-AUC</th>
                    <th>Quadratic Weighted Cohen's κ</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><span class="kvp-tag-pill kvp-tag-blue">5-Class Grading</span></td>
                    <td><b>DenseNet121 + OARSI aux heads</b> (CORAL, 3-seed mean)</td>
                    <td><b style="color:#38bdf8">{acc_5c_str}</b></td>
                    <td><span style="color:#64748b">—</span></td>
                    <td><span class="kvp-tag-pill kvp-tag-green">{qwk_str}</span></td>
                </tr>
                <tr>
                    <td><span class="kvp-tag-pill kvp-tag-green">Binary OA Detection</span></td>
                    <td><b>ConvNeXt-Small</b> (Best Single)</td>
                    <td><b style="color:#34d399">87.0%</b></td>
                    <td><b style="color:#38bdf8">0.9457</b></td>
                    <td><span class="kvp-tag-pill kvp-tag-green">0.7571 (val)</span></td>
                </tr>
                <tr>
                    <td><span class="kvp-tag-pill kvp-tag-green">Binary OA Detection</span></td>
                    <td><b>DenseNet + ConvNeXt</b> Ensemble</td>
                    <td><b style="color:#34d399">86.5%</b></td>
                    <td><b style="color:#38bdf8">0.9463</b></td>
                    <td><span style="color:#64748b">—</span></td>
                </tr>
                <tr>
                    <td><span class="kvp-tag-pill kvp-tag-purple">Grouped (2-Group)</span></td>
                    <td>X-ray model, regrouped (3-seed mean)</td>
                    <td><b style="color:#c084fc">{acc_gb_str}</b></td>
                    <td><b style="color:#38bdf8">{auc_gb_str}</b></td>
                    <td><span style="color:#64748b">—</span></td>
                </tr>
                <tr>
                    <td><span class="kvp-tag-pill kvp-tag-purple">Grouped (3-Group)</span></td>
                    <td>X-ray model, regrouped (3-seed mean)</td>
                    <td><b style="color:#c084fc">{acc_g3_str}</b></td>
                    <td><span style="color:#64748b">—</span></td>
                    <td><span style="color:#64748b">—</span></td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption("κ marked (val) is validation κ at training time; full evaluation confusion matrices live in Model Performance. "
               "Exact 5-class accuracy is limited by adjacent-grade ambiguity, so quadratic κ and within-one-grade accuracy are the primary measures.")


def page_diagnosis(device, image_models):
    st.subheader("🩻 X-ray Diagnosis")
    st.caption("Upload a single knee crop or a full bilateral AP standing radiograph — automatic knee detection, dual-joint KL grading, and OA screening.")

    # Top options: Upload Mode + Model Selector
    c_mode, c_model = st.columns([1.3, 1.0])
    with c_mode:
        input_mode = st.radio(
            "Radiograph Input Mode",
            ["🔄 Auto-Detect (Single or Bilateral)", "👥 Bilateral AP (Both Knees: R & L)", "🩻 Single Knee (Focused Crop)"],
            horizontal=True,
            help="Auto-Detect analyzes aspect ratio and bone separation to automatically split bilateral views into Right and Left knee joints.",
        )
    with c_model:
        model_choice = st.selectbox("Model", list(image_models.keys()) + [ENSEMBLE_OPTION], key="diag_model_choice")

    if "diag_sample_choice" not in st.session_state:
        st.session_state["diag_sample_choice"] = None

    # File uploader and quick sample loader
    c_up, c_sample = st.columns([2.2, 1.3])
    with c_up:
        uploaded = st.file_uploader(
            "Upload a knee radiograph",
            type=["png", "jpg", "jpeg", "bmp"],
            key="diagnosis_xray_upload",
        )
    with c_sample:
        st.caption("🧪 Quick Test Samples:")
        s_c1, s_c2 = st.columns(2)
        with s_c1:
            if st.button("👥 Bilateral", help="Load sample 2-in-1 standing AP radiograph"):
                st.session_state["diag_sample_choice"] = "bilateral"
        with s_c2:
            if st.button("🦾 Prosthesis", help="Load knee arthroplasty / TKA prosthesis sample"):
                st.session_state["diag_sample_choice"] = "prosthesis"
        if st.button("🦴 Native Knee", help="Load single native knee crop"):
            st.session_state["diag_sample_choice"] = "native"

    active_image = None
    if uploaded is not None:
        active_image = Image.open(uploaded).convert("RGB")
        st.session_state["diag_sample_choice"] = None
    elif st.session_state.get("diag_sample_choice") == "bilateral" and DEMO_BILATERAL_PATH.exists():
        active_image = Image.open(DEMO_BILATERAL_PATH).convert("RGB")
    elif st.session_state.get("diag_sample_choice") == "prosthesis" and DEMO_PROSTHESIS_PATH.exists():
        active_image = Image.open(DEMO_PROSTHESIS_PATH).convert("RGB")
    elif st.session_state.get("diag_sample_choice") == "native" and DEMO_NATIVE_PATH.exists():
        active_image = Image.open(DEMO_NATIVE_PATH).convert("RGB")

    if active_image is None:
        st.info("Upload an X-ray above or click one of the **Quick Test Samples** to start.")
        return

    # Decide single vs bilateral
    is_bilateral = False
    details = {}
    if "Auto-Detect" in input_mode:
        is_bilateral, conf, details = detect_bilateral_radiograph(active_image)
        if is_bilateral:
            st.success(f"👥 **Bilateral AP Radiograph Detected** (Both knees detected · Aspect ratio {details['aspect_ratio']:.2f} · {conf:.0%} confidence). Splitting into Patient Right (R) and Left (L) knee joints.")
        else:
            st.info("🩻 **Single Knee Radiograph Detected** (Focused single joint).")
    elif "Bilateral" in input_mode:
        is_bilateral = True
        _, _, details = detect_bilateral_radiograph(active_image)
    else:
        is_bilateral = False

    models = ensemble_models(image_models) if model_choice == ENSEMBLE_OPTION else [image_models[model_choice]]
    bin_model, bin_name = load_binary_model()

    if is_bilateral:
        # Fine-tune settings
        with st.expander("⚙️ Bilateral Split & Joint Centering Settings", expanded=False):
            fc1, fc2 = st.columns(2)
            with fc1:
                def_split = int(details.get("split_ratio", 0.5) * 100)
                split_pct = st.slider("Split Divider (% from left)", 35, 65, def_split, 1,
                                      help="Adjust the divider line between Right and Left knee joints.")
            with fc2:
                joint_focus = st.checkbox("Focus Joint Space (Anatomical Centering)", value=True,
                                          help="Crops joint-sized windows centred on the joint line to match model training, "
                                               "and grades at three scales.")
            joint_pct = st.slider("Joint line height (% from top of knee)", 30, 70, int(DEFAULT_JOINT_Y * 100), 1,
                                  help="Move until the cyan/purple box is centred on the joint space (the dark line "
                                       "between femur and tibia).")

        rk, lk, annotated = extract_bilateral_knees(
            active_image,
            split_ratio=split_pct / 100.0,
            joint_focus=joint_focus,
            joint_y_frac=joint_pct / 100.0,
        )

        st.image(annotated, caption="Bilateral Radiograph Overview: Right Knee (R, Cyan) · Left Knee (L, Purple)", width="stretch")

        # Run inference on both knees
        # Joint-sized windows at three scales with contrast normalisation: a single wide crop
        # makes the grade depend on crop scale, not anatomy.
        if joint_focus:
            half_r, half_l = split_bilateral_halves(active_image, split_ratio=split_pct / 100.0)
            reports = [
                multiscale_report(knee_windows(half, joint_pct / 100.0),
                                  lambda w: image_probs(w, models, device))
                for half in (half_r, half_l)
            ]
            rep_r, rep_l = reports
            probs_r, probs_l = rep_r.probs, rep_l.probs
        else:
            rep_r = rep_l = None
            probs_r = image_probs(rk, models, device)
            probs_l = image_probs(lk, models, device)
        pred_r, conf_r = int(probs_r.argmax()), float(probs_r.max())
        pred_l, conf_l = int(probs_l.argmax()), float(probs_l.max())

        # Automated heuristic check for metallic prostheses (TKA)
        is_pr, _, _ = detect_knee_prosthesis(rk)
        is_pl, _, _ = detect_knee_prosthesis(lk)

        col_r, col_l = st.columns(2, gap="large")
        with col_r:
            st.markdown("### 🦵 Patient Right Knee (R)")
            st.image(rk, caption=f"Right Knee Joint Crop · {rk.width}×{rk.height}px", width="stretch")

            r_type = st.radio(
                "Joint Condition (Right Knee)",
                ["🦴 Native Biological Joint", "🦾 Post-TKA / Prosthesis (Replacement)"],
                index=1 if is_pr else 0,
                key="r_knee_joint_type",
                horizontal=True,
            )

            if "Post-TKA" in r_type:
                st.markdown(prosthesis_card("Right Knee — Arthroplasty Assessment"), unsafe_allow_html=True)
            else:
                st.markdown(grade_card(f"Right Knee — {model_choice}", pred_r, conf_r, "#38bdf8"), unsafe_allow_html=True)
                if bin_model is not None:
                    p_oa_r = binary_oa_prob(bin_model, rk, device)
                    verdict_r = ('<span class="chip-warn">OA detected</span>'
                                 if p_oa_r >= 0.5 else '<span class="chip-ok">No OA</span>')
                    body_r = (f"<div style='font-size:2rem;font-weight:800;color:#38bdf8'>{p_oa_r:.1%}</div>"
                              f"<div class='kvp-sub'>probability of OA (KL ≥ 2)</div>"
                              f"<div class='kvp-bar'><div class='kvp-fill' style='width:{int(p_oa_r*100)}%'></div></div>"
                              f"<div style='margin-top:.6rem'>{verdict_r}</div>")
                    st.markdown(card(f"OA screening — {bin_name}", body_r), unsafe_allow_html=True)
                if rep_r is not None and rep_r.scale_sensitive:
                    st.warning("⚠️ **Low-reliability grade** — confidence is under 70% or the grade changes with crop scale "
                               "(per-scale grades: " + ", ".join(f"KL {g}" for g in rep_r.per_scale_pred)
                               + f"; mean confidence {rep_r.confidence:.0%}). On held-out test images, predictions below 70% "
                               "confidence were exactly right only about half the time (usually within one grade). "
                               "Re-centre the joint line above, or treat this as indeterminate and have a clinician review the film.")
                st.markdown("**Right Knee Class Probabilities**")
                st.bar_chart(pd.Series(probs_r, index=[f"G{i} {KL_LABELS[i]}" for i in range(5)]), height=200)

        with col_l:
            st.markdown("### 🦵 Patient Left Knee (L)")
            st.image(lk, caption=f"Left Knee Joint Crop · {lk.width}×{lk.height}px", width="stretch")

            l_type = st.radio(
                "Joint Condition (Left Knee)",
                ["🦴 Native Biological Joint", "🦾 Post-TKA / Prosthesis (Replacement)"],
                index=1 if is_pl else 0,
                key="l_knee_joint_type",
                horizontal=True,
            )

            if "Post-TKA" in l_type:
                st.markdown(prosthesis_card("Left Knee — Arthroplasty Assessment"), unsafe_allow_html=True)
            else:
                st.markdown(grade_card(f"Left Knee — {model_choice}", pred_l, conf_l, "#a855f7"), unsafe_allow_html=True)
                if bin_model is not None:
                    p_oa_l = binary_oa_prob(bin_model, lk, device)
                    verdict_l = ('<span class="chip-warn">OA detected</span>'
                                 if p_oa_l >= 0.5 else '<span class="chip-ok">No OA</span>')
                    body_l = (f"<div style='font-size:2rem;font-weight:800;color:#a855f7'>{p_oa_l:.1%}</div>"
                              f"<div class='kvp-sub'>probability of OA (KL ≥ 2)</div>"
                              f"<div class='kvp-bar'><div class='kvp-fill' style='width:{int(p_oa_l*100)}%;background:#a855f7'></div></div>"
                              f"<div style='margin-top:.6rem'>{verdict_l}</div>")
                    st.markdown(card(f"OA screening — {bin_name}", body_l), unsafe_allow_html=True)
                if rep_l is not None and rep_l.scale_sensitive:
                    st.warning("⚠️ **Low-reliability grade** — confidence is under 70% or the grade changes with crop scale "
                               "(per-scale grades: " + ", ".join(f"KL {g}" for g in rep_l.per_scale_pred)
                               + f"; mean confidence {rep_l.confidence:.0%}). On held-out test images, predictions below 70% "
                               "confidence were exactly right only about half the time (usually within one grade). "
                               "Re-centre the joint line above, or treat this as indeterminate and have a clinician review the film.")
                st.markdown("**Left Knee Class Probabilities**")
                st.bar_chart(pd.Series(probs_l, index=[f"G{i} {KL_LABELS[i]}" for i in range(5)]), height=200)

        # Comparative Summary Card
        r_is_tka = "Post-TKA" in r_type
        l_is_tka = "Post-TKA" in l_type

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        if r_is_tka and l_is_tka:
            sym_title = "🦾 Bilateral Total Knee Arthroplasty (Both Knees Replaced)"
            sym_desc = "Both knee joints have undergone surgical arthroplasty (TKA). Kellgren-Lawrence osteoarthritis grading is clinically inapplicable to both limbs."
            meta_html = "<div><b>Right (R):</b> TKA Prosthesis (KL N/A)</div><div><b>Left (L):</b> TKA Prosthesis (KL N/A)</div>"
            dominant_grade = 0
            rehab_context = "Bilateral total knee arthroplasty with artificial hardware in both limbs."
        elif r_is_tka or l_is_tka:
            tka_side = "Right Knee (R)" if r_is_tka else "Left Knee (L)"
            native_side = "Left Knee (L)" if r_is_tka else "Right Knee (R)"
            native_pred = pred_l if r_is_tka else pred_r
            native_conf = conf_l if r_is_tka else conf_r

            sym_title = "⚠️ Unilateral Knee Arthroplasty (TKA) with Contralateral Native OA"
            sym_desc = (
                f"Patient has undergone <b>Total Knee Replacement on the {tka_side}</b>, while the contralateral "
                f"<b>{native_side} is a native biological joint presenting with KL {native_pred} ({KL_LABELS[native_pred]})</b>. "
                f"Primary clinical strategy: conservative physical therapy and joint offloading to preserve the native {native_side} "
                f"and prevent or delay secondary surgery, while supporting low-impact mobility for the replaced {tka_side}."
            )
            meta_html = (
                f"<div><b>{tka_side}:</b> TKA Prosthesis (KL N/A)</div>"
                f"<div><b>{native_side}:</b> Native KL {native_pred} ({native_conf:.1%})</div>"
            )
            dominant_grade = native_pred
            rehab_context = f"Unilateral TKA on {tka_side}. Contralateral {native_side} has native KL {native_pred} ({KL_LABELS[native_pred]})."
        elif pred_r == pred_l:
            sym_title = "⚖️ Symmetric Bilateral Findings"
            sym_desc = f"Both native knees exhibit concordant radiographic severity: <b>KL Grade {pred_r} ({KL_LABELS[pred_r]})</b>."
            meta_html = f"<div><b>Right (R):</b> KL {pred_r} ({conf_r:.1%})</div><div><b>Left (L):</b> KL {pred_l} ({conf_l:.1%})</div>"
            dominant_grade = pred_r
            rehab_context = f"Bilateral AP radiograph: Right knee KL {pred_r}, Left knee KL {pred_l}."
        else:
            diff = abs(pred_r - pred_l)
            more_severe = "Right Knee (R)" if pred_r > pred_l else "Left Knee (L)"
            sym_title = f"⚠️ Asymmetric Bilateral Findings ({diff}-Grade Discrepancy)"
            sym_desc = (f"Asymmetric disease detected: <b>Right Knee is KL {pred_r} ({KL_LABELS[pred_r]})</b>, "
                        f"while <b>Left Knee is KL {pred_l} ({KL_LABELS[pred_l]})</b>. "
                        f"Primary intervention should focus on the more affected <b>{more_severe}</b>.")
            meta_html = f"<div><b>Right (R):</b> KL {pred_r} ({conf_r:.1%})</div><div><b>Left (L):</b> KL {pred_l} ({conf_l:.1%})</div>"
            dominant_grade = max(pred_r, pred_l)
            rehab_context = f"Bilateral AP radiograph: Right knee KL {pred_r}, Left knee KL {pred_l}."

        st.markdown(
            card(
                sym_title,
                f"<div class='kvp-sub' style='font-size:1.05rem;color:#f8fafc;'>{sym_desc}</div>"
                f"<div style='margin-top:0.8rem;display:flex;gap:1.5rem;'>{meta_html}</div>",
            ),
            unsafe_allow_html=True,
        )

        # Rehab plan
        rehab_title = "Post-Arthroplasty & Joint Preservation Plan" if (r_is_tka or l_is_tka) else f"Recommended Bilateral Plan (KL {dominant_grade} - {KL_LABELS[dominant_grade]})"
        with st.expander(f"🏃 {rehab_title}", expanded=False):
            recommender = load_rehab_recommender()
            rehab_res = recommender.recommend(kl_grade=dominant_grade, patient_context=rehab_context)
            st.markdown(rehab_res.synthesis)
            st.caption(f"⚠️ {rehab_res.disclaimer}")
            show_rehab_sources(rehab_res)

    else:
        # Standard Single Knee
        col_img, col_results = st.columns([1, 1], gap="large")
        with col_img:
            st.image(
                active_image,
                caption=f"Uploaded Single Knee X-ray · {active_image.width} × {active_image.height}px",
                width="stretch",
            )
        with col_results:
            is_single_pr, _, _ = detect_knee_prosthesis(active_image)
            single_type = st.radio(
                "Joint Condition",
                ["🦴 Native Biological Joint", "🦾 Post-TKA / Prosthesis (Replacement)"],
                index=1 if is_single_pr else 0,
                key="single_knee_joint_type",
                horizontal=True,
            )

            if "Post-TKA" in single_type:
                st.markdown(prosthesis_card("Knee Radiograph — Arthroplasty Assessment"), unsafe_allow_html=True)
            else:
                probs = image_probs(active_image, models, device)
                pred, confidence = int(probs.argmax()), float(probs.max())

                st.markdown("<div style='margin-top: 0.9rem;'></div>", unsafe_allow_html=True)
                st.markdown(grade_card(f"KL grade — {model_choice}", pred, confidence, "#38bdf8"), unsafe_allow_html=True)
                st.caption("Grades: 0 Normal · 1 Doubtful · 2 Mild · 3 Moderate · 4 Severe")

                if bin_model is not None:
                    p_oa = binary_oa_prob(bin_model, active_image, device)
                    verdict = ('<span class="chip-warn">OA detected</span>'
                               if p_oa >= 0.5 else '<span class="chip-ok">No OA</span>')
                    body = (f"<div style='font-size:2rem;font-weight:800;color:#38bdf8'>{p_oa:.1%}</div>"
                            f"<div class='kvp-sub'>probability of OA (KL ≥ 2)</div>"
                            f"<div class='kvp-bar'><div class='kvp-fill' style='width:{int(p_oa*100)}%'></div></div>"
                            f"<div style='margin-top:.6rem'>{verdict}</div>")
                    st.markdown(card(f"OA screening — {bin_name}", body), unsafe_allow_html=True)
                    st.caption("Dedicated binary detector (KL 0-1 vs 2-4), threshold 0.50.")
                else:
                    st.markdown(card("OA screening", "<div class='kvp-sub'>No binary checkpoint found.</div>"), unsafe_allow_html=True)

                st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
                st.markdown("**Class Probabilities**")
                st.bar_chart(pd.Series(probs, index=[f"G{i} {KL_LABELS[i]}" for i in range(5)]), height=220)

                with st.expander(f"🏃 Recommended Rehabilitation Plan for KL {pred} ({KL_LABELS[pred]})", expanded=False):
                    recommender = load_rehab_recommender()
                    rehab_res = recommender.recommend(kl_grade=pred, patient_context=f"Radiographic diagnosis: KL Grade {pred} ({KL_LABELS[pred]}).")
                    st.markdown(rehab_res.synthesis)
                    st.caption(f"⚠️ {rehab_res.disclaimer}")
                    show_rehab_sources(rehab_res)


def page_xai(device, image_models):
    st.subheader("🔥 Explainability")
    st.caption("Grad-CAM / Score-CAM / LIME show where the model's attention was strongest (DenseNet backbones). These maps are indicative visual aids, not a diagnosis.")

    if "xai_sample_choice" not in st.session_state:
        st.session_state["xai_sample_choice"] = None

    c_x1, c_x2 = st.columns([2.2, 1.3])
    with c_x1:
        xai_uploaded = st.file_uploader(
            "Upload a knee X-ray for explanation",
            type=["png", "jpg", "jpeg", "bmp"],
            key="xai_upload",
        )
    with c_x2:
        st.caption("🧪 Quick Test Samples:")
        xs_c1, xs_c2 = st.columns(2)
        with xs_c1:
            if st.button("👥 Bilateral", key="xai_bilat_btn", help="Load sample 2-in-1 radiograph"):
                st.session_state["xai_sample_choice"] = "bilateral"
        with xs_c2:
            if st.button("🦴 Single", key="xai_native_btn", help="Load single knee crop"):
                st.session_state["xai_sample_choice"] = "native"

    active_xai_img = None
    if xai_uploaded is not None:
        active_xai_img = Image.open(xai_uploaded).convert("RGB")
        st.session_state["xai_sample_choice"] = None
    elif st.session_state.get("xai_sample_choice") == "bilateral" and DEMO_BILATERAL_PATH.exists():
        active_xai_img = Image.open(DEMO_BILATERAL_PATH).convert("RGB")
    elif st.session_state.get("xai_sample_choice") == "native" and DEMO_NATIVE_PATH.exists():
        active_xai_img = Image.open(DEMO_NATIVE_PATH).convert("RGB")

    if active_xai_img is None:
        st.info("Upload an X-ray to generate heatmaps or click one of the **Quick Test Samples**.")
        return

    # Check if bilateral
    is_bilateral, _, details = detect_bilateral_radiograph(active_xai_img)
    xai_method = st.radio("Explanation method", ["Grad-CAM", "Score-CAM", "LIME"], horizontal=True)

    xai_choices = [k for k in image_models if k.startswith("DenseNet121")] or list(image_models)[:1]
    xai_label = st.selectbox("Model to explain (DenseNet121 backbones)", xai_choices, key="xai_model_choice")
    model = image_models[xai_label]
    xai_tf = model_tf(model)
    _res = load_app_results()
    if _res:
        _x = _res["xai"]
        st.caption(
            f"Measured on validation images (insertion / deletion AUC; insertion higher is better, deletion lower is better): "
            f"Grad-CAM {_x['gradcam']['insertion_auc']:.2f} / {_x['gradcam']['deletion_auc']:.2f}, "
            f"Score-CAM {_x['scorecam']['insertion_auc']:.2f} / {_x['scorecam']['deletion_auc']:.2f}, "
            f"LIME {_x['lime']['insertion_auc']:.2f} / {_x['lime']['deletion_auc']:.2f}, "
            f"random {_x['random']['insertion_auc']:.2f} / {_x['random']['deletion_auc']:.2f}. "
            "Above chance on insertion; only marginally better than chance on deletion.")

    if is_bilateral:
        st.info("👥 **Bilateral Radiograph Detected** · Visualizing explainability heatmaps on anatomical knee crops.")
        joint_sel = st.radio("Joint to explain", ["Both Knees (Side-by-Side)", "Right Knee (R)", "Left Knee (L)"], horizontal=True)

        rk, lk, _ = extract_bilateral_knees(active_xai_img, joint_focus=True)

        def _explain_joint(img):
            if xai_method == "Grad-CAM":
                p, c, ov = gradcam_explain(model, img, xai_tf, device)
            elif xai_method == "Score-CAM":
                p, c, ov = scorecam_explain(model, img, xai_tf, device)
            else:
                p, c, imp, _ = lime_explain(model, img, xai_tf, device)
                from kneevision.xai.base import overlay_heatmap
                img_np = np.array(img.resize((224, 224)))
                ov = overlay_heatmap(imp, img_np, alpha=0.55)
            return p, c, ov

        with st.spinner(f"Computing {xai_method} explanation..."):
            if joint_sel == "Both Knees (Side-by-Side)":
                pr, cr, ov_r = _explain_joint(rk)
                pl, cl, ov_l = _explain_joint(lk)

                col_r, col_l = st.columns(2, gap="large")
                with col_r:
                    st.markdown("### 🦵 Patient Right Knee (R)")
                    st.image(ov_r, caption=f"{xai_method} Heatmap · Predicted KL {pr} ({KL_LABELS[pr]}) · {cr:.1%}", width="stretch")
                    st.markdown(
                        card("Right Knee Feature Saliency",
                             f"<div class='kvp-sub' style='color:#f8fafc;'>Predicted <b>KL Grade {pr}</b> ({KL_LABELS[pr]})</div>"
                             f"<div class='kvp-sub' style='margin-top:0.5rem;color:#94a3b8;'>Regions with the strongest model attention; indicative only.</div>"),
                        unsafe_allow_html=True,
                    )
                with col_l:
                    st.markdown("### 🦵 Patient Left Knee (L)")
                    st.image(ov_l, caption=f"{xai_method} Heatmap · Predicted KL {pl} ({KL_LABELS[pl]}) · {cl:.1%}", width="stretch")
                    st.markdown(
                        card("Left Knee Feature Saliency",
                             f"<div class='kvp-sub' style='color:#f8fafc;'>Predicted <b>KL Grade {pl}</b> ({KL_LABELS[pl]})</div>"
                             f"<div class='kvp-sub' style='margin-top:0.5rem;color:#94a3b8;'>Regions with the strongest model attention; indicative only.</div>"),
                        unsafe_allow_html=True,
                    )
                return
            elif joint_sel == "Right Knee (R)":
                active_xai_img = rk
            else:
                active_xai_img = lk

    with st.spinner(f"Computing {xai_method} explanation..."):
        if xai_method == "Grad-CAM":
            pred, conf, overlay = gradcam_explain(model, active_xai_img, xai_tf, device)
        elif xai_method == "Score-CAM":
            pred, conf, overlay = scorecam_explain(model, active_xai_img, xai_tf, device)
        else:
            pred, conf, importance, _segments = lime_explain(model, active_xai_img, xai_tf, device)
            from kneevision.xai.base import overlay_heatmap
            img_np = np.array(active_xai_img.resize((224, 224)))
            overlay = overlay_heatmap(importance, img_np, alpha=0.55)
            # Add subtle grid lines on the superpixels
            grid_size = 7
            cell_h, cell_w = 224 // grid_size, 224 // grid_size
            for i in range(1, grid_size):
                overlay[i * cell_h : i * cell_h + 1, :] = [255, 255, 255]
                overlay[:, i * cell_w : i * cell_w + 1] = [255, 255, 255]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(active_xai_img, caption="Analyzed Knee Crop", width="stretch")
    with c2:
        kind = f"{xai_method} Superpixel Heatmap" if xai_method == "LIME" else f"{xai_method} Heatmap"
        st.image(overlay, caption=f"{kind} — predicted KL {pred}", width="stretch")
    with c3:
        if xai_method == "LIME":
            expl_detail = "Warmer superpixels are patches that most increased the predicted grade when present in this perturbation test. Indicative only."
        else:
            expl_detail = "Warmer areas are where the model's attention was strongest for this prediction. They do not by themselves identify specific radiographic findings."
        st.markdown(
            card(
                "How to read this",
                f"<div class='kvp-sub' style='font-size:1.05rem;color:#f8fafc;'>Predicted <b>KL Grade {pred}</b> ({KL_LABELS[pred]}) &middot; {conf:.1%} conf</div>"
                f"<div class='kvp-sub' style='margin-top:0.6rem;color:#94a3b8;line-height:1.5;'>{expl_detail}</div>",
            ),
            unsafe_allow_html=True,
        )


def page_clinical(device):
    st.subheader("📝 Clinical Report Diagnosis")
    st.caption("Test the text model with a sample report or paste your own. No image is required.")

    clinical_model, is_trained = load_clinical_model()
    if clinical_model is None:
        st.error("Clinical model could not be loaded.")
        return
    if not is_trained:
        st.warning("⚠️ No trained checkpoint (models/best_clinical.pt) — outputs below come from a "
                   "random-initialized network and are placeholders. Train with scripts/train_clinical.py.")

    st.markdown(
        card("How this works",
             "<div class='flow-row'><div class='flow-node'>&#128221; Report text</div>"
             "<div class='flow-arrow'>&rarr;</div>"
             "<div class='flow-node'>BERT tokenizer<small>max 512 tokens</small></div>"
             "<div class='flow-arrow'>&rarr;</div>"
             "<div class='flow-node'>Bio_ClinicalBERT<small>[CLS] embedding</small></div>"
             "<div class='flow-arrow'>&rarr;</div>"
             "<div class='flow-node'>Classifier head</div>"
             "<div class='flow-arrow'>&rarr;</div>"
             "<div class='flow-node output'>KL grade 0&ndash;4</div></div>"),
        unsafe_allow_html=True,
    )
    st.write("")
    st.info(
        "The sample reports omit explicit KL labels to avoid giving the model the answer. "
        "They are synthetic smoke tests, not held-out evaluation data."
    )
    if "clinical_report_input" not in st.session_state:
        st.session_state.clinical_report_input = ""
    st.selectbox(
        "Optional sample input",
        list(CLINICAL_EXAMPLES),
        key="clinical_example_select",
        help="Choose a synthetic report to fill the text box. Sample IDs do not reveal the reference category before prediction.",
        on_change=_load_selected_example,
        args=("clinical_example_select", "clinical_report_input", CLINICAL_EXAMPLES),
    )
    report = st.text_area(
        "Radiology report text",
        key="clinical_report_input",
        placeholder="FINDINGS: Definite medial joint-space narrowing with marginal osteophytes...",
        height=170,
    )
    if st.button("Predict KL grade", type="primary"):
        if not report.strip():
            st.warning("Enter a report first.")
        else:
            preds, confs = clinical_model.predict([report], device)
            pred, conf = preds[0], confs[0]
            probs = clinical_probs(clinical_model, report, device)
            b1, b2 = st.columns([1, 1.3])
            with b1:
                st.markdown(grade_card("Text-based diagnosis", int(pred), conf, "#7c3aed"), unsafe_allow_html=True)
            with b2:
                st.bar_chart(pd.Series(probs, index=[f"G{i} {KL_LABELS[i]}" for i in range(5)]), height=230)
            sample_grade = CLINICAL_EXAMPLE_REFERENCE.get(st.session_state.clinical_example_select)
            if sample_grade is not None:
                st.caption(
                    f"Illustrative sample reference: KL {sample_grade} ({KL_LABELS[sample_grade]}). "
                    "This synthetic example is not a validation case; agreement does not measure model accuracy."
                )

            # Rehabilitation Protocol
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            recommender = load_rehab_recommender()
            with st.spinner(f"Synthesizing personalized rehabilitation protocol (AAOS/OARSI guidelines + {recommender.llm.label})..."):
                rehab_result = recommender.recommend(kl_grade=int(pred), patient_context=report)

            badge = f"🧠 {load_rehab_recommender().llm.label} · RAG" if rehab_result.used_llm else "📄 Guideline Excerpts (Direct)"
            badge_style = "kvp-tag-purple" if rehab_result.used_llm else "kvp-tag-blue"

            st.markdown(
                f"""
                <div class="kvp-card" style="margin-bottom: 0.8rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:0.75rem; margin-bottom:1rem; flex-wrap:wrap; gap:0.5rem;">
                        <div>
                            <h4 style="margin:0 !important; color:#f8fafc !important; font-size:1.1rem !important;">🏃 Tailored Rehabilitation Protocol</h4>
                            <div style="color:#94a3b8; font-size:0.82rem; margin-top:0.2rem;">Evidence-based protocol personalized for <strong>KL {int(pred)} ({KL_LABELS[int(pred)]})</strong></div>
                        </div>
                        <span class="kvp-tag-pill {badge_style}">{badge}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div style='background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1.1rem 1.3rem;margin-bottom:1rem;line-height:1.65;font-size:0.95rem;color:#e2e8f0;'>"
                f"{rehab_result.synthesis}"
                f"</div>",
                unsafe_allow_html=True,
            )

            show_rehab_sources(rehab_result)

            st.caption(f"⚠️ {rehab_result.disclaimer}")


def page_fusion(device, image_models):
    st.subheader("🔀 Multimodal Fusion")
    st.caption("Test image and report inputs together. Preset X-rays and reports are illustrative examples, not patient-matched records.")

    clinical_model, _clinical_trained = load_clinical_model()
    fusion_model, fusion_trained = load_fusion_model()
    fusion_flow_diagram()
    st.write("")

    fusion_presets = [
        "Manual input",
        "Example 0 · No definite OA features",
        "Example 1 · Tiny osteophyte, preserved space",
        "Example 2 · Osteophytes, possible narrowing",
        "Example 3 · Definite narrowing and sclerosis",
        "Example 4 · Near-complete joint-space loss",
    ]
    fusion_reports = {name: DEMO_REPORTS[grade] for grade, name in enumerate(fusion_presets[1:])}
    if "fusion_report_input" not in st.session_state:
        st.session_state.fusion_report_input = ""
    preset_choice = st.selectbox(
        "⚡ Quick Preset Cases",
        fusion_presets,
        key="fusion_example_select",
        on_change=_load_selected_example,
        args=("fusion_example_select", "fusion_report_input", fusion_reports),
    )

    preset_img_path = None
    demo_dict = demo_images()
    if "Example 0" in preset_choice:
        preset_img_path = demo_dict.get(0)
    elif "Example 1" in preset_choice:
        preset_img_path = demo_dict.get(1)
    elif "Example 2" in preset_choice:
        preset_img_path = demo_dict.get(2)
    elif "Example 3" in preset_choice:
        preset_img_path = demo_dict.get(3)
    elif "Example 4" in preset_choice:
        preset_img_path = demo_dict.get(4)

    if preset_choice != "Manual input" and preset_img_path is None:
        st.warning("No reference X-ray is available for this example in data/raw/test. Upload an X-ray to test both modalities.")

    col_img, col_txt = st.columns(2)
    with col_img:
        up = st.file_uploader("1 · Knee X-ray (AP view)", type=["png", "jpg", "jpeg", "bmp"], key="fusion_img")
        if up is not None:
            active_img = Image.open(up).convert("RGB")
            st.image(active_img, caption="Active X-ray Image", width=240)
        elif preset_img_path:
            active_img = Image.open(preset_img_path).convert("RGB")
            st.image(active_img, caption="Preset Demo Image", width=240)
        else:
            active_img = None

    with col_txt:
        report = st.text_area(
            "2 · Clinical / Radiology Report",
            key="fusion_report_input",
            placeholder="FINDINGS: ... IMPRESSION: ...",
            height=180,
        )

    mode_col, slider_col = st.columns([1.6, 1])
    with mode_col:
        fusion_mode = st.radio(
            "Fusion Mechanism",
            ["🧠 Deep Neural Fusion (Trained Multimodal Head)", "🎚️ Dynamic Heuristic Blend (α slider)"],
            horizontal=True,
            help="Deep Neural Fusion feeds visual & textual embeddings through the trained late fusion network.",
        )

    alpha = 0.5
    if "Dynamic" in fusion_mode:
        with slider_col:
            alpha = st.slider("Fusion weight α (text influence)", 0.0, 1.0, 0.5, 0.05,
                              help="0 = image only, 1 = text only")

    ready_img = active_img is not None
    ready_txt = bool(report.strip()) and clinical_model is not None

    if st.button("Run Multimodal Diagnosis", type="primary", disabled=not (ready_img or ready_txt)):
        img_p = txt_p = None
        if ready_img:
            img_p = image_probs(active_img, ensemble_models(image_models), device)
        if ready_txt:
            txt_p = clinical_probs(clinical_model, report, device)

        if "Deep Neural" in fusion_mode and fusion_model is not None and ready_img and ready_txt:
            _, _, fused = fusion_model.predict(active_img, report, device)
            badge_title = "🧠 Deep Neural Fusion"
            badge_color = "#0f766e"
        elif img_p is not None and txt_p is not None:
            fused = (1 - alpha) * img_p + alpha * txt_p
            badge_title = f"⊕ Heuristic Blend (α={alpha:.2f})"
            badge_color = "#b45309"
        else:
            fused = img_p if img_p is not None else txt_p
            badge_title = "Single Modality Active"
            badge_color = "#0369a1"

        ipred = int(img_p.argmax()) if img_p is not None else None
        tpred = int(txt_p.argmax()) if txt_p is not None else None
        fpred, fconf = int(fused.argmax()), float(fused.max())

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(grade_card("🩻 Image CNN Branch", ipred, float(img_p[ipred]) if ipred is not None else None, "#0284c7"),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(grade_card("📝 Clinical BERT Branch", tpred, float(txt_p[tpred]) if tpred is not None else None, "#7c3aed"),
                        unsafe_allow_html=True)
        with c3:
            st.markdown(grade_card(badge_title, fpred, fconf, badge_color), unsafe_allow_html=True)

        st.markdown("##### 📊 Comparative Probability Distribution across Modalities")
        chart_data = {
            "Image CNN": img_p if img_p is not None else np.zeros(5),
            "Clinical BERT": txt_p if txt_p is not None else np.zeros(5),
            "Multimodal Decision": fused,
        }
        st.bar_chart(pd.DataFrame(chart_data, index=[f"KL {i} ({KL_LABELS[i]})" for i in range(5)]), height=300)

        # Evidence-Based Rehabilitation Guidance (RAG)
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        recommender = load_rehab_recommender()
        with st.spinner(f"Synthesizing personalized rehabilitation protocol (AAOS/OARSI guidelines + {recommender.llm.label})..."):
            rehab_result = recommender.recommend(kl_grade=fpred, patient_context=report)

        badge = f"🧠 {load_rehab_recommender().llm.label} · RAG" if rehab_result.used_llm else "📄 Guideline Excerpts (Direct)"
        badge_style = "kvp-tag-purple" if rehab_result.used_llm else "kvp-tag-blue"

        st.markdown(
            f"""
            <div class="kvp-card" style="margin-bottom: 0.8rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:0.75rem; margin-bottom:1rem; flex-wrap:wrap; gap:0.5rem;">
                    <div>
                        <h4 style="margin:0 !important; color:#f8fafc !important; font-size:1.1rem !important;">🏃 Tailored Rehabilitation Protocol</h4>
                        <div style="color:#94a3b8; font-size:0.82rem; margin-top:0.2rem;">Evidence-based regimen synthesized for <strong>KL {fpred} ({KL_LABELS[fpred]})</strong> and clinical report findings</div>
                    </div>
                    <span class="kvp-tag-pill {badge_style}">{badge}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1.1rem 1.3rem;margin-bottom:1rem;line-height:1.65;font-size:0.95rem;color:#e2e8f0;'>"
            f"{rehab_result.synthesis}"
            f"</div>",
            unsafe_allow_html=True,
        )

        if not rehab_result.used_llm:
            st.caption("ℹ️ Local LLM (Ollama) was not detected at localhost:11434; displaying direct evidence guidelines.")

        show_rehab_sources(rehab_result)

        st.caption(f"⚠️ {rehab_result.disclaimer}")

        if "Deep Neural" in fusion_mode and not fusion_trained:
            st.warning("`models/best_fusion.pt` was not detected. Train via `scripts/train_fusion.py` to enable trained neural weights.")
    elif not ready_img and not ready_txt:
        st.info("Provide a Knee X-ray and/or clinical report above to run multimodal evaluation.")



def page_showcase():
    """Results by task - interactive clinical framing strategies."""
    import json

    st.markdown(
        "<div style=\"background:linear-gradient(120deg,#064e3b 0%,#0369a1 60%,#312e81 100%);"
        "border-radius:18px;padding:1.6rem 2rem;margin-bottom:1.2rem;\">"
        "<h2 style=\"color:#f0fdf4;margin:0 0 .3rem 0;font-size:1.7rem;\">&#127942; Results by Task</h2>"
        "<p style=\"color:#a7f3d0;margin:0;font-size:.97rem;\">"
        "Accuracy depends on the question asked and on whether clinical findings are available. "
        "Every number below uses the identical 1,656-sample held-out test set; strategies that set aside uncertain or "
        "doubtful cases report their coverage, and fusion results use clinical findings that define the KL grade."
        "</p></div>",
        unsafe_allow_html=True,
    )

    showcase_json = REPORTS_DIR / "showcase" / "model_showcase_summary.json"
    showcase_png  = REPORTS_DIR / "showcase" / "model_showcase_comparison.png"

    if not showcase_json.exists():
        with st.spinner("Running showcase benchmark (first-time only, ~2 min on GPU) ..."):
            import os
            import subprocess
            env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent / "src")}
            result = subprocess.run(
                ["python", str(Path(__file__).parent / "scripts" / "showcase_benchmark.py")],
                capture_output=True, text=True, env=env, check=False,
            )
            if result.returncode != 0:
                st.error("Benchmark failed. Run manually:\n```\nuv run python scripts/showcase_benchmark.py\n```")
                st.code(result.stderr[-3000:], language="text")
                return

    if not showcase_json.exists():
        st.info("No showcase results yet. Generate them with:\n```\nuv run python scripts/showcase_benchmark.py\n```")
        return

    with open(showcase_json) as f:
        results = json.load(f)

    STRATEGY_COLORS = {
        "baseline":          "#64748b",
        "binary":            "#0ea5e9",
        "grouped":           "#8b5cf6",
        "confidence_gating": "#10b981",
        "definitive":        "#f59e0b",
    }
    STRATEGY_ICONS = {
        "baseline":          "&#x1F9E0;",
        "binary":            "&#x1F535;",
        "grouped":           "&#x1F3E5;",
        "confidence_gating": "&#x1F3AF;",
        "definitive":        "&#x2B50;",
    }
    STRATEGY_NAMES = {
        "baseline":          "Baseline 5-Class Fusion",
        "binary":            "Binary OA Triage",
        "grouped":           "3-Tier Actionability",
        "confidence_gating": "Confidence Gating",
        "definitive":        "Definitive Grading",
    }

    # Top metric cards
    top_keys = ["definitive_grading", "binary_oa_triage", "clinical_3tier",
                "confidence_75", "multimodal_5class"]
    top_keys = [k for k in top_keys if k in results]

    cols = st.columns(len(top_keys))
    for col, key in zip(cols, top_keys):
        r = results[key]
        acc       = r["accuracy"] * 100
        strat     = r["strategy"]
        icon      = STRATEGY_ICONS.get(strat, "&#x1F4CA;")
        name      = r["label"].replace("\n", " ")
        is_90     = acc >= 90.0
        badge_col = "#10b981" if is_90 else "#f59e0b"
        above_badge = (
            "<div style=\"background:#064e3b;color:#6ee7b7;border-radius:999px;"
            "font-size:.72rem;padding:.1rem .5rem;margin-top:.4rem;display:inline-block;\">"
            "&#10003; Above 90%</div>"
        ) if is_90 else ""
        with col:
            st.markdown(
                f"<div style=\"background:#1e293b;border:1px solid #334155;border-radius:14px;"
                f"padding:.9rem 1rem;text-align:center;\">"
                f"<div style=\"font-size:1.6rem;\">{icon}</div>"
                f"<div style=\"font-size:1.85rem;font-weight:900;color:{badge_col};line-height:1.1;\">"
                f"{acc:.1f}%</div>"
                f"<div style=\"color:#94a3b8;font-size:.78rem;margin-top:.25rem;\">{name}</div>"
                f"{above_badge}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    tab_chart, tab_interactive, tab_details, tab_context = st.tabs([
        "&#x1F4CA; Comparison Chart", "&#x1F39A;&#xFE0F; Interactive Explorer",
        "&#x1F4CB; Strategy Details", "&#x1F52C; Clinical Context"
    ])

    # Tab 1: Static comparison chart
    with tab_chart:
        if showcase_png.exists():
            st.image(str(showcase_png),
                     caption="KneeVision++ Accuracy Across Clinical Framing Strategies",
                     width="stretch")
        else:
            st.warning("Chart not generated yet. Run `scripts/showcase_benchmark.py`.")
        st.caption(
            "Left: Accuracy of each strategy on 1,656 test-set samples. "
            "Middle: Coverage vs. accuracy trade-off for confidence gating. "
            "Right: Breakdown of automated vs. radiologist-deferred cases (P >= 0.75 scenario)."
        )

    # Tab 2: Interactive explorer
    with tab_interactive:
        st.markdown("#### Interactive Confidence Threshold Explorer")
        st.caption(
            "Drag the slider to see how tightening the confidence threshold "
            "increases accuracy (fewer automated cases, higher per-case precision)."
        )

        gate_entries = sorted(
            [(r["threshold"], r["accuracy"], r["coverage"])
             for r in results.values() if r.get("strategy") == "confidence_gating"],
            key=lambda x: x[0],
        )
        base_acc = results["multimodal_5class"]["accuracy"]
        gate_entries = [(0.50, base_acc, 1.0)] + gate_entries
        thresholds = [g[0] for g in gate_entries]

        sel_idx = st.select_slider(
            "Minimum model confidence required for auto-resolution:",
            options=list(range(len(gate_entries))),
            value=1 if len(gate_entries) > 1 else 0,
            format_func=lambda i: "No gating" if thresholds[i] == 0.50 else f"P >= {thresholds[i]:.2f}",
        )
        sel_th, sel_acc, sel_cov = gate_entries[sel_idx]
        sel_deferred = 1.0 - sel_cov

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("5-Class Accuracy", f"{sel_acc:.2%}",
                      delta=f"+{(sel_acc - base_acc)*100:.1f}pp vs baseline" if sel_idx > 0 else None)
        with m2:
            st.metric("Auto-Resolved Cases", f"{sel_cov:.1%}",
                      delta=f"-{(1.0-sel_cov)*100:.1f}% deferred" if sel_idx > 0 else "100% (all cases)")
        with m3:
            st.metric("Radiologist Queue", f"{sel_deferred:.1%}",
                      help="Cases below confidence threshold, routed for specialist review.")

        bar_keys   = list(results.keys())
        bar_labels = [r["label"].replace("\n", " ") for r in results.values()]
        bar_accs   = [r["accuracy"] * 100 for r in results.values()]
        bar_strats = [r["strategy"] for r in results.values()]
        highlight_key = f"confidence_{int(sel_th*100)}" if sel_th != 0.50 else "multimodal_5class"
        bar_colors = [
            STRATEGY_COLORS.get(s, "#64748b") if k != highlight_key else "#087e8b"
            for k, s in zip(bar_keys, bar_strats)
        ]

        try:
            import plotly.graph_objects as go
            fig_bar = go.Figure(go.Bar(
                y=bar_labels[::-1], x=bar_accs[::-1], orientation="h",
                marker_color=bar_colors[::-1],
                text=[f"{a:.1f}%" for a in bar_accs[::-1]],
                textposition="outside", textfont_color="#172b3a",
            ))
            fig_bar.add_vline(x=90, line_dash="dash", line_color="#f43f5e",
                              annotation_text="90% target", annotation_font_color="#f43f5e")
            fig_bar.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#f7fafb",
                font_color="#345367", height=420,
                xaxis={"range": [0, 108], "ticksuffix": "%", "gridcolor": "#dce7ee"},
                yaxis={"tickfont_size": 10},
                margin={"l": 10, "r": 30, "t": 20, "b": 10},
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        except ImportError:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.set_facecolor("#f7fafb")
            fig2.patch.set_facecolor("#ffffff")
            ax2.barh(bar_labels[::-1], bar_accs[::-1], color=bar_colors[::-1])
            ax2.axvline(90, color="#f43f5e", linestyle="--")
            ax2.tick_params(colors="#526779")
            ax2.set_xlabel("Accuracy (%)", color="#345367")
            for s in ax2.spines.values():
                s.set_edgecolor("#dce7ee")
            st.pyplot(fig2)
            plt.close(fig2)

    # Tab 3: Strategy details
    with tab_details:
        st.markdown("#### Per-Strategy Clinical Framing Details")
        for key, r in results.items():
            strat = r["strategy"]
            acc   = r["accuracy"] * 100
            cov   = r["coverage"] * 100
            color = STRATEGY_COLORS.get(strat, "#64748b")
            label = r["label"].replace("\n", " ")
            with st.expander(f"{label}  |  {acc:.1f}% accuracy, {cov:.0f}% coverage",
                             expanded=(acc >= 94.0)):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f"<div style=\"background:#1e293b;border-left:4px solid {color};"
                        f"padding:.8rem 1rem;border-radius:0 10px 10px 0;\">"
                        f"<b style=\"color:#f8fafc\">Accuracy:</b> "
                        f"<span style=\"color:{color};font-size:1.3rem;font-weight:900\">{acc:.1f}%</span><br>"
                        f"<b style=\"color:#f8fafc\">Coverage:</b> {cov:.0f}% of test cases<br>"
                        f"<b style=\"color:#f8fafc\">Strategy:</b> {STRATEGY_NAMES.get(strat,'')}</div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f"<div style=\"color:#94a3b8;padding:.5rem 0\">{r['description']}</div>",
                        unsafe_allow_html=True,
                    )

    # Tab 4: Clinical context
    with tab_context:
        render_context_tab()


def render_context_tab():
    st.markdown("#### Why accuracy numbers vary with the question asked")
    res = load_app_results()
    if not res:
        st.info("docs/app_results.json not found. Run `uv run python scripts/build_app_results.py`.")
        return
    bt = res["branch_tasks"]
    pm = lambda v: f"{v[0]:.1f} ± {v[1]:.1f}%" if isinstance(v, list) else f"{v:.1f}%"  # noqa: E731
    st.table(pd.DataFrame([
        {"Task": "Binary (No OA KL0–1 vs OA KL2–4)", "X-ray only": pm(bt["xray_final"]["binary"]), "Clinical text": pm(bt["clinical"]["binary"]), "Fusion": pm(bt["fusion"]["binary"])},
        {"Task": "3 groups (KL0–1 / KL2 / KL3–4)", "X-ray only": pm(bt["xray_final"]["three_class"]), "Clinical text": pm(bt["clinical"]["three_class"]), "Fusion": pm(bt["fusion"]["three_class"])},
        {"Task": "5 grades (KL0–KL4)", "X-ray only": pm(bt["xray_final"]["five_class"]), "Clinical text": pm(bt["clinical"]["five_class"]), "Fusion": pm(bt["fusion"]["five_class"])},
    ]))
    st.info(
        "**Coarser questions give higher accuracy** because most mistakes are between neighbouring grades. "
        "Quadratic weighted κ and within-one-grade accuracy are the main measures for 5-grade KL grading."
    )
    st.warning(
        "**Fusion and clinical-text results are not image-only.** The clinical text contains per-compartment radiographic "
        "findings (joint-space narrowing, osteophytes, sclerosis) that define the KL grade, so those branches score higher "
        "partly because of that overlap. The X-ray-only column is the image-only result."
    )


def render_final_results():
    res = load_app_results()
    if not res:
        st.info("docs/app_results.json not found. Run `uv run python scripts/build_app_results.py`.")
        return
    st.markdown("**Accuracy by task and branch** (frozen test set, 1,656 radiographs from 828 patients)")
    render_context_tab()
    st.markdown("**Model comparison on the 5-grade task** (same training recipe; mean ± SD over seeds where more than one)")
    f2 = lambda v, d=3: f"{v[0]:.{d}f} ± {v[1]:.{d}f}" if v[1] else f"{v[0]:.{d}f}"  # noqa: E731
    st.dataframe(pd.DataFrame([
        {"Model": m["model"], "Seeds": m["seeds"], "Accuracy %": f2(m["accuracy"], 1), "Macro-F1": f2(m["macro_f1"]), "AUC": f2(m["auc"]),
         "QWK": f2(m["qwk"]), "MAE": f2(m["mae"]), "Within-1 %": f2(m["within1"], 1), "Params (M)": f"{m['params_m']:.1f}"}
        for m in res["models"]
    ]), hide_index=True, width="stretch")
    st.caption("Single-seed rows differ from DenseNet121 by less than the seed-to-seed spread (about 0.013 QWK); treat small gaps as ties.")
    left, right = st.columns(2)
    with left:
        st.markdown("**How often each grade is found** (recall, best model, 3-seed mean)")
        st.bar_chart(pd.DataFrame({"Recall %": res["per_class_recall_final"]}, index=[f"KL{k}" for k in range(5)]))
    with right:
        c = res["complexity"]
        st.markdown("**Model size**")
        st.markdown(f"- **{c['params_m']:.1f} M** parameters, **{c['ckpt_mb']:.0f} MB**, **{c['gflops']:.2f} GFLOPs**  \n"
                    f"- about {c['ms_per_image_b32_fp32']:.1f} ms per image (batch 32, laptop GPU; hardware specific)")
    ex = res["external_shift"]
    st.warning(
        "**Known limitation: performance falls at a new hospital.** On an independent cohort "
        f"({ex['cohort']}), the plain CORAL model's zero-shot agreement was QWK {ex['plain_zero_shot']['qwk'][0]:.2f} "
        f"(vs about 0.82 internally), with a tendency to over-call disease. Re-estimating batch-norm statistics on unlabelled images "
        f"gave QWK {ex['plain_adabn']['qwk'][0]:.2f}, and re-fitting thresholds on 50 labelled images {ex['plain_recal50']['qwk'][0]:.2f}. "
        f"Domain-robust training raised zero-shot QWK from {ex['baseline40_zero_shot']['qwk'][0]:.2f} to {ex['robust40_zero_shot']['qwk'][0]:.2f} "
        "(matched 40-epoch runs, 2 seeds). A new site needs local validation before any use.")
    st.caption("Research prototype evaluated retrospectively; not a diagnostic device. Numbers come from docs/app_results.json.")


def page_performance():
    st.subheader("📊 Model Performance")
    st.caption("All numbers read live from models/ metadata, reports/, and mlflow.db.")
    perf_final, perf_5c, perf_fusion, perf_bin, perf_grp = st.tabs([
        "Final Results", "5-Class Grading", "Multimodal Fusion", "Binary OA Detection", "Grouped Evaluation"
    ])

    with perf_final:
        render_final_results()

    with perf_5c:
        left, right = st.columns([1, 1])
        with left:
            rows = [
                {"Model": m["model_name"], "Best val κ": round(m["best_kappa"], 4) if m.get("best_kappa") is not None else None,
                 "Epoch": m.get("epoch"), "Image": f"{m.get('image_size', 224)}px", "Checkpoint": m["file"]}
                for m in checkpoint_metas() if not m.get("binary") and "clinical" not in m["file"]
            ]
            if rows:
                st.markdown("**Trained checkpoints (5-class)**")
                st.table(pd.DataFrame(rows))
            img = report_image("evaluate")
            if img:
                st.image(img, caption="Confusion matrix — test set", width="stretch")
        with right:
            st.markdown("**Classification report — test set**")
            st.text(report_text("evaluate"))

    with perf_fusion:
        fleft, fright = st.columns([1.1, 0.9])
        with fleft:
            st.markdown("**Modality Comparison (Image vs Clinical Text vs Fusion)**")
            comp_img = REPORTS_DIR / "evaluate_fusion" / "modality_comparison.png"
            if comp_img.exists():
                st.image(str(comp_img), caption="Modality Benchmark Comparison", width="stretch")
            else:
                st.info("Run `uv run python scripts/evaluate_fusion.py` to generate fusion benchmark charts.")

        with fright:
            st.markdown("**Multimodal Classification Report**")
            st.text(report_text("evaluate_fusion"))
            cm_fusion = report_image("evaluate_fusion")
            if cm_fusion:
                st.image(cm_fusion, caption="Multimodal Fusion Confusion Matrix", width="stretch")

    with perf_bin:
        left, right = st.columns([1, 1])
        with left:
            rows = [
                {"Model": m["model_name"], "Best val κ": round(m["best_kappa"], 4) if m.get("best_kappa") is not None else None,
                 "Epoch": m.get("epoch"), "Image": f"{m.get('image_size', 224)}px", "Checkpoint": m["file"]}
                for m in checkpoint_metas() if m.get("binary")
            ]
            if rows:
                st.markdown("**Trained checkpoints (OA vs No OA)**")
                st.table(pd.DataFrame(rows))
            img = report_image("binary_dedicated")
            if img:
                st.image(img, caption="Confusion matrix — latest binary evaluation", width="stretch")
        with right:
            st.markdown("**Classification report — latest binary ensemble**")
            st.text(report_text("binary_dedicated"))

    with perf_grp:
        st.table(pd.DataFrame([
            {"Strategy": label,
             "Test accuracy": f"{report_accuracy(name):.1%}" if report_accuracy(name) else "—",
             "Report": name}
            for name, label in [
                ("grouped_binary", "2-group (KL 0-1 vs 2-4)"),
                ("grouped_3class", "3-group (0-1 / 2-3 / 4)"),
            ]
        ]))
        gcol1, gcol2 = st.columns(2)
        for col, name, label in [
            (gcol1, "grouped_binary", "Binary grouping (KL 0-1 vs 2-4)"),
            (gcol2, "grouped_3class", "3-class grouping (0-1 / 2-3 / 4)"),
        ]:
            with col:
                st.markdown(f"**{label}**")
                img = report_image(name)
                if img:
                    st.image(img, caption="Confusion matrix — test set", width="stretch")
                st.text(report_text(name))


def page_demo(device, image_models):
    st.subheader("🧪 Guided Demo — walk through every KL grade")
    st.caption("Compare a held-out X-ray with a synthetic, label-free report example in the same grade category. They are not patient-matched; model predictions may differ from the reference grade.")

    available = demo_images()
    if not available:
        st.error("No demo images found under data/raw/test/{0..4}.")
        return

    grades = sorted(available)
    fmt = {g: f"KL {g} · {KL_LABELS[g]}" for g in grades}
    grade = st.select_slider("Choose KL grade", options=grades, format_func=lambda g: fmt[g], value=grades[len(grades) // 2])
    st.info(f"**{KL_LABELS[grade]}** — {GRADE_FEATURES[grade]}")

    col_img, col_txt, col_pred = st.columns([1.05, 1.15, 1.1])

    with col_img:
        st.markdown("**🩻 Reference X-ray**")
        image = Image.open(available[grade]).convert("RGB")
        st.image(image, caption=f"Held-out test image · reference category KL {grade}", width="stretch")

    with col_txt:
        st.markdown("**📝 Matching radiology report (demo)**")
        st.text_area("Report", value=DEMO_REPORTS[grade], height=250, key=f"demo_report_{grade}",
                     label_visibility="collapsed")

    with col_pred:
        st.markdown("**🤖 Model predictions on these inputs**")

        img_p = image_probs(image, ensemble_models(image_models), device)
        ipred, iconf = int(img_p.argmax()), float(img_p.max())

        bin_model, bin_name = load_binary_model()
        p_oa = binary_oa_prob(bin_model, image, device) if bin_model is not None else None

        clinical_model, trained = load_clinical_model()
        txt_p = txt_pred = None
        if clinical_model is not None:
            txt_p = clinical_probs(clinical_model, DEMO_REPORTS[grade], device)
            txt_pred = int(txt_p.argmax())

        hit = "✅ correct" if ipred == grade else "❌ off by one" if abs(ipred - grade) == 1 else "❌ miss"
        st.markdown(
            card(f"CNN ensemble — {hit}",
                 f"<div class='kvp-sub'>predicts <b>KL {ipred} ({KL_LABELS[ipred]})</b> @ {iconf:.0%}</div>"),
            unsafe_allow_html=True,
        )
        if p_oa is not None:
            verdict = ('<span class="chip-warn">OA detected</span>' if p_oa >= 0.5
                       else '<span class="chip-ok">No OA</span>')
            expect = grade >= 2
            agree = (p_oa >= 0.5) == expect
            st.markdown(card(f"OA screening ({bin_name}) — {'✅' if agree else '↔'}",
                             f"<div class='kvp-sub'>P(OA) = <b>{p_oa:.0%}</b> &nbsp;{verdict}</div>"),
                        unsafe_allow_html=True)
        if txt_pred is not None and not trained:
            st.markdown(card("Clinical BERT",
                             "<div class='kvp-sub'>untrained demo weights — see 📝 page</div>"),
                        unsafe_allow_html=True)

    with st.expander("Per-grade class probabilities (image ensemble)", expanded=False):
        st.bar_chart(pd.Series(img_p, index=[f"G{i} {KL_LABELS[i]}" for i in range(5)]), height=240)


def page_rehab():
    st.subheader("🏃 Evidence-Based Rehabilitation Guidance (RAG)")
    st.caption("Choose a reference KL grade and optional symptom context. The output summarizes retrieved public guidelines; it is educational, not a prescription.")

    col_kl, col_ctx = st.columns([1, 1.8], gap="medium")
    with col_kl:
        grade = st.select_slider(
            "Kellgren–Lawrence (KL) Grade",
            options=[0, 1, 2, 3, 4],
            format_func=lambda g: f"KL {g} · {KL_LABELS[g]}",
            value=2,
        )
    with col_ctx:
        if "rehab_context_input" not in st.session_state:
            st.session_state.rehab_context_input = ""
        st.selectbox(
            "Load an example symptom profile",
            list(REHAB_EXAMPLES),
            key="rehab_example_select",
            on_change=_load_selected_example,
            args=("rehab_example_select", "rehab_context_input", REHAB_EXAMPLES),
        )
        custom_input = st.text_input(
            "Patient Clinical Profile",
            key="rehab_context_input",
            placeholder="e.g. persistent pain on stairs, reduced walking tolerance",
        )

    _llm = load_rehab_recommender().llm
    st.caption(f"🤖 Language model for the summary: **{_llm.label}**"
               + ("" if getattr(_llm, "provider", "ollama") != "ollama" else
                  " · to use Grok, set `XAI_API_KEY` in `.streamlit/secrets.toml` (local) or Streamlit Cloud → Secrets"))
    if getattr(_llm, "provider", "ollama") != "ollama":
        st.caption("🔒 The text above is sent to an external AI provider to write the summary. "
                   "Do not enter names, dates of birth or other identifying details.")

    if st.button("Generate Rehabilitation Plan", type="primary"):
        recommender = load_rehab_recommender()
        with st.spinner(f"Retrieving orthopedic guidelines & synthesizing plan via {recommender.llm.label}..."):
            result = recommender.recommend(grade, patient_context=custom_input)

        badge = f"🧠 {recommender.llm.label} · RAG" if result.used_llm else "📄 Guideline Excerpts (Direct)"
        badge_style = "kvp-tag-purple" if result.used_llm else "kvp-tag-blue"

        st.markdown(
            f"""
            <div class="kvp-card" style="margin-top: 1.2rem; margin-bottom: 0.5rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:0.75rem; margin-bottom:1rem; flex-wrap:wrap; gap:0.5rem;">
                    <h4 style="margin:0 !important; color:#f8fafc !important; font-size:1rem !important;">🏃 Tailored Rehabilitation Protocol</h4>
                    <span class="kvp-tag-pill {badge_style}">{badge} &middot; KL {grade} ({KL_LABELS[grade]})</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(result.synthesis)

        if not result.used_llm:
            st.warning("The language model wasn't available (or its summary failed the faithfulness check) — "
                       "showing retrieved guideline excerpts directly instead of an LLM-synthesized summary.")

        show_rehab_sources(result)

        st.caption(f"⚠️ {result.disclaimer}")


def page_mlflow():
    import sqlite3

    from mlflow.exceptions import MlflowException

    st.subheader("📈 MLflow Tracking")
    try:
        df = mlflow_runs_table()
    except (MlflowException, OSError, sqlite3.Error):
        # No mlflow.db shipped with this deployment (e.g. a fresh clone) — the sqlite
        # backend has no schema yet, which mlflow surfaces as a query error, not an empty result.
        df = pd.DataFrame()
    if df.empty:
        st.info("No MLflow runs found in mlflow.db. Train a model to populate tracking.")
        return
    st.dataframe(df, width="stretch", hide_index=True)
    st.markdown("Open the full MLflow UI at [http://localhost:5000](http://localhost:5000) "
                "(start it with `uv run python scripts/mlflow_server.py server --port 5000`).")


# ----------------------------------------------------------------------------
# App shell
# ----------------------------------------------------------------------------

PAGES = {
    "🏠 Overview": page_overview,
    "🏆 Results by Task": page_showcase,
    "🩻 X-ray Diagnosis": lambda: page_diagnosis(*load_image_models()),
    "🔥 Explainability": lambda: page_xai(*load_image_models()),
    "📝 Clinical Text": lambda: page_clinical(load_image_models()[0]),
    "🔀 Multimodal Fusion": lambda: page_fusion(*load_image_models()),
    "🧪 Guided Demo": lambda: page_demo(*load_image_models()),
    "🏃 Rehab Recommendation": page_rehab,
    "📊 Performance": page_performance,
    "📈 MLflow": page_mlflow,
}

PAGE_GROUPS = {
    "Start here": ["🏠 Overview", "🧪 Guided Demo"],
    "Test models": ["🩻 X-ray Diagnosis", "📝 Clinical Text", "🔀 Multimodal Fusion", "🔥 Explainability"],
    "Rehabilitation": ["🏃 Rehab Recommendation"],
    "Results": ["🏆 Results by Task", "📊 Performance", "📈 MLflow"],
}


def main():
    st.set_page_config(page_title="KneeVision++ | Multimodal OA Diagnosis", page_icon="🦵", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            """
            <div class="kvp-sidebar-header">
                <div class="kvp-brand">
                    <div class="kvp-logo-badge">🦵</div>
                    <div>
                        <div class="kvp-brand-title">KneeVision++</div>
                        <div class="kvp-brand-tag">RESEARCH AI &middot; v2.4</div>
                    </div>
                </div>
                <div class="kvp-brand-desc">Multimodal Knee OA Diagnosis<br><span>KL 0–4 &middot; CNN + BERT &middot; Explainable</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        nav_group = st.selectbox("Go to", list(PAGE_GROUPS), key="nav_group")
        page = st.radio(
            "Page",
            PAGE_GROUPS[nav_group],
            label_visibility="collapsed",
            key=f"nav_page_{nav_group}",
        )
        st.markdown(
            """
            <div class="kvp-sidebar-footer">
                <div class="kvp-footer-status"><span class="dot-online"></span> Testing path</div>
                <div class="kvp-footer-meta">1. Guided Demo: image + example report</div>
                <div class="kvp-footer-meta">2. Clinical Text: report only</div>
                <div class="kvp-footer-meta">3. Rehab: grade + symptoms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _device, image_models = load_image_models()
    if not image_models:
        st.error("No loadable image checkpoints found in models/. Train a model first.")
        return
    PAGES[page]()


if __name__ == "__main__":
    main()
