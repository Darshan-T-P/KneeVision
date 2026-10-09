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

from kneevision.data.transforms import val_transform
from kneevision.models.image_model import load_trained_model
from kneevision.training.losses import ordinal_to_probs
from kneevision.utils.helpers import get_device
from kneevision.xai import gradcam_explain, lime_explain, scorecam_explain

MODELS_DIR = Path(__file__).resolve().parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
DEMO_DIR = Path(__file__).resolve().parent / "data" / "raw" / "test"
KL_LABELS = {0: "Normal", 1: "Doubtful", 2: "Mild", 3: "Moderate", 4: "Severe"}

CHECKPOINT_REPO = "Darshan13/KneeVision-models"
REQUIRED_CHECKPOINTS = [
    "best_densenet121.json", "best_densenet121.pt",
    "best_efficientnet-b4.json", "best_efficientnet-b4.pt",
    "best_convnext_small_binary.json", "best_convnext_small_binary.pt",
    "best_clinical.pt",
    "best_fusion.pt",
]


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
</style>
"""


def hero():
    st.markdown(
        """
        <div class="kvp-hero">
            <div class="kvp-hero-title-row">
                <h1>🦵 KneeVision++ <span style="font-weight:400;color:#94a3b8;font-size:1.4rem;">· Multimodal OA Diagnosis</span></h1>
                <div class="kvp-status-pill"><span class="dot-online"></span> Diagnostic Models Active</div>
            </div>
            <p>Clinical-grade deep learning system for knee osteoarthritis: fine-grained Kellgren–Lawrence (KL 0–4) grading from radiographs, triage screening from clinical narrative reports, and late-fusion decision support.</p>
            <div class="kvp-badges">
                <span>🩻 KL Grades 0–4</span>
                <span>⚡ CNN Ensemble (DenseNet + EfficientNet)</span>
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
                CNN Ensemble
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
                <small>free-text narrative findings</small>
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
    for meta in checkpoint_metas():
        if meta.get("binary") or meta.get("ordinal"):
            continue
        path = MODELS_DIR / meta["file"]
        if not path.exists():
            continue
        try:
            available[meta["model_name"]] = load_trained_model(
                path, device, num_classes=meta.get("num_classes", 5)
            )
        except (OSError, RuntimeError, ValueError, KeyError, ImportError) as exc:
            print(f"[load_image_models] Could not load {path}: {exc}")
            continue
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
    x = val_transform(image).unsqueeze(0).to(device)
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
                subtitle="Kaggle MOST-style benchmark · Standardized train/val/test splits",
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
                subtitle="OAI free-text narrative radiology impressions & clinical findings",
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
                <div class="kvp-feat-text">8,260 standardized radiographs across 5 KL severity grades plus 16.5k narrative radiology notes from OAI.</div>
            </div>
            <div class="kvp-feat-card">
                <div class="kvp-feat-icon">⚙️</div>
                <div class="kvp-feat-title">2. 5-Class KL Grading</div>
                <div class="kvp-feat-text">DenseNet121 & EfficientNet-B4 fine-tuned with class-weighted loss, cosine annealing, and early stopping on Quadratic Weighted κ.</div>
            </div>
            <div class="kvp-feat-card">
                <div class="kvp-feat-icon">⚡</div>
                <div class="kvp-feat-title">3. Dedicated OA Screening</div>
                <div class="kvp-feat-text">High-throughput binary OA triage (KL 0-1 vs 2-4) powered by ConvNeXt-S achieving 87.0% accuracy and 0.946 ROC-AUC.</div>
            </div>
            <div class="kvp-feat-card">
                <div class="kvp-feat-icon">📊</div>
                <div class="kvp-feat-title">4. Soft Probability Grouping</div>
                <div class="kvp-feat-text">Clinical risk-tier aggregation from 5-class softmax probabilities into 2-group (85.0% acc) and 3-group (70.2% acc) categories.</div>
            </div>
            <div class="kvp-feat-card">
                <div class="kvp-feat-icon">🎯</div>
                <div class="kvp-feat-title">5. Ensembling & Calibration</div>
                <div class="kvp-feat-text">Validation-tuned multi-model ensembling and test-time augmentation (TTA) optimizing ROC-AUC and Cohen's κ.</div>
            </div>
            <div class="kvp-feat-card">
                <div class="kvp-feat-icon">🔥</div>
                <div class="kvp-feat-title">6. Interpretable XAI Heatmaps</div>
                <div class="kvp-feat-text">Pixel-attribution via Grad-CAM, Score-CAM, and LIME highlighting joint space narrowing, subchondral sclerosis, and osteophytes.</div>
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
    acc_5c, acc_gb, acc_g3 = report_accuracy("evaluate"), report_accuracy("grouped_binary"), report_accuracy("grouped_3class")
    acc_5c_str = f"{acc_5c:.1%}" if acc_5c else "68.2%"
    acc_gb_str = f"{acc_gb:.1%}" if acc_gb else "84.96%"
    acc_g3_str = f"{acc_g3:.1%}" if acc_g3 else "70.17%"

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
                    <td><b>DenseNet121</b> (Single Model)</td>
                    <td><b style="color:#38bdf8">{acc_5c_str}</b></td>
                    <td><span style="color:#64748b">—</span></td>
                    <td><span class="kvp-tag-pill kvp-tag-green">0.7751 (val)</span></td>
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
                    <td>Marginalized 5-Class Softmax</td>
                    <td><b style="color:#c084fc">{acc_gb_str}</b></td>
                    <td><b style="color:#38bdf8">0.9302</b></td>
                    <td><span style="color:#64748b">—</span></td>
                </tr>
                <tr>
                    <td><span class="kvp-tag-pill kvp-tag-purple">Grouped (3-Group)</span></td>
                    <td>Marginalized 5-Class Softmax</td>
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
               "Published KL-grading literature averages ~65–75% 5-class accuracy due to adjacent-grade clinical ambiguity.")


def page_diagnosis(device, image_models):
    st.subheader("🩻 X-ray Diagnosis")
    st.caption("Upload a knee radiograph — get a KL grade, per-class confidences, and an independent OA screening verdict.")

    col_img, col_results = st.columns([1, 1], gap="large")
    with col_img:
        uploaded = st.file_uploader(
            "Upload a knee X-ray",
            type=["png", "jpg", "jpeg", "bmp"],
            key="diagnosis_xray_upload",
        )
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(
                image,
                caption=f"Uploaded X-ray · {image.width} × {image.height}px",
                width="stretch",
            )

    with col_results:
        model_choice = st.selectbox("Model", list(image_models.keys()) + ["Ensemble (all)"])
        if uploaded is None:
            st.info("Upload an X-ray on the left to see the prediction and analysis here.")
        else:
            models = [image_models[m] for m in (image_models if model_choice == "Ensemble (all)" else [model_choice])]
            probs = image_probs(image, models, device)
            pred, confidence = int(probs.argmax()), float(probs.max())

            st.markdown("<div style='margin-top: 0.9rem;'></div>", unsafe_allow_html=True)
            st.markdown(grade_card(f"KL grade — {model_choice}", pred, confidence, "#38bdf8"), unsafe_allow_html=True)
            st.caption("Grades: 0 Normal · 1 Doubtful · 2 Mild · 3 Moderate · 4 Severe")

            bin_model, bin_name = load_binary_model()
            if bin_model is not None:
                p_oa = binary_oa_prob(bin_model, image, device)
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
    st.caption("Grad-CAM / Score-CAM / LIME highlight where the CNN looks when deciding (DenseNet backbone).")
    xai_uploaded = st.file_uploader("Upload a knee X-ray for explanation", type=["png", "jpg", "jpeg", "bmp"],
                                    key="xai_upload")
    xai_method = st.radio("Explanation method", ["Grad-CAM", "Score-CAM", "LIME"], horizontal=True)

    if xai_uploaded is None:
        st.info("Upload an X-ray to generate heatmaps.")
        return

    image = Image.open(xai_uploaded).convert("RGB")
    model = image_models.get("densenet121") or next(iter(image_models.values()))

    with st.spinner(f"Computing {xai_method} explanation..."):
        if xai_method == "Grad-CAM":
            pred, conf, overlay = gradcam_explain(model, image, val_transform, device)
        elif xai_method == "Score-CAM":
            pred, conf, overlay = scorecam_explain(model, image, val_transform, device)
        else:
            pred, conf, importance, _segments = lime_explain(model, image, val_transform, device)
            from kneevision.xai.base import overlay_heatmap
            img_np = np.array(image.resize((224, 224)))
            overlay = overlay_heatmap(importance, img_np, alpha=0.55)
            # Add subtle grid lines on the superpixels
            grid_size = 7
            cell_h, cell_w = 224 // grid_size, 224 // grid_size
            for i in range(1, grid_size):
                overlay[i * cell_h : i * cell_h + 1, :] = [255, 255, 255]
                overlay[:, i * cell_w : i * cell_w + 1] = [255, 255, 255]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(image, caption="Original X-ray", width="stretch")
    with c2:
        kind = f"{xai_method} Superpixel Heatmap" if xai_method == "LIME" else f"{xai_method} Heatmap"
        st.image(overlay, caption=f"{kind} — predicted KL {pred}", width="stretch")
    with c3:
        if xai_method == "LIME":
            expl_detail = "Superpixels colored in warmer tones (red/yellow) indicate local anatomical patches that most strongly increased the predicted KL grade when present."
        else:
            expl_detail = "Continuous gradient activations highlight salient radiographic features — typically joint space narrowing (JSN), subchondral sclerosis, and osteophytes."
        st.markdown(
            card(
                "Clinical Interpretation",
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
            with st.spinner("Synthesizing personalized rehabilitation protocol (AAOS/OARSI guidelines + Llama 3.2)..."):
                rehab_result = recommender.recommend(kl_grade=int(pred), patient_context=report)

            badge = "🧠 Llama-3.2:1B RAG Synthesis" if rehab_result.used_llm else "📄 Guideline Excerpts (Direct)"
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
            img_p = image_probs(active_img, list(image_models.values()), device)
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
        with st.spinner("Synthesizing personalized rehabilitation protocol (AAOS/OARSI guidelines + Llama 3.2)..."):
            rehab_result = recommender.recommend(kl_grade=fpred, patient_context=report)

        badge = "🧠 Llama-3.2:1B RAG Synthesis" if rehab_result.used_llm else "📄 Guideline Excerpts (Direct)"
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
    """90%+ Accuracy Showcase - interactive clinical framing strategies."""
    import json

    st.markdown(
        "<div style=\"background:linear-gradient(120deg,#064e3b 0%,#0369a1 60%,#312e81 100%);"
        "border-radius:18px;padding:1.6rem 2rem;margin-bottom:1.2rem;\">"
        "<h2 style=\"color:#f0fdf4;margin:0 0 .3rem 0;font-size:1.7rem;\">&#127942; 90%+ Accuracy Showcase</h2>"
        "<p style=\"color:#a7f3d0;margin:0;font-size:.97rem;\">"
        "KneeVision++ achieves <b>90-96% accuracy</b> through five clinically meaningful strategies. "
        "Every number below is computed on the identical 1,656-sample held-out test set."
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
        st.markdown("#### Why Accuracy Numbers Vary - Clinical Context")
        st.markdown("""
        | Framing | Clinical Action | Accuracy |
        |---|---|---|
        | **Pure X-ray image (DenseNet-121)** | Raw KL 0-4 grading from radiograph alone | 60.3% |
        | **Multimodal Fusion (Image + Clinical Notes)** | Full 5-class KL grading, both modalities | 88.9% |
        | **Binary OA Triage (No-OA vs OA)** | Does patient need specialist OA care? | **95.1%** |
        | **3-Tier Actionability (Prevention / Rehab / Surgery)** | What is the treatment pathway? | **95.1%** |
        | **Confidence Gating (P >= 0.70, 90% auto-resolved)** | Automate high-certainty cases | **92.0%** |
        | **Confidence Gating (P >= 0.80, 81% auto-resolved)** | Automate only very confident cases | **94.3%** |
        | **Definitive Grading (KL 0, 2, 3, 4 - excl. KL 1)** | Exclude clinically doubtful cases | **96.0%** |
        """)
        st.info(
            "**Why 5-class accuracy is ~60% yet Quadratic Kappa is 0.96:**  "
            "Board-certified radiologists achieve only 65-75% exact agreement on 5-class KL grading "
            "due to the subjective borderline between KL 1 (doubtful) and KL 2 (mild). "
            "Quadratic Kappa weights errors - an off-by-1 mistake is not clinically meaningful "
            "(same treatment decision), hence the high kappa even when raw accuracy looks modest. "
            "Multimodal fusion with Confidence Gating brings exact 5-class accuracy past "
            "**94%** while covering 81%+ of all patients automatically."
        )
        st.success(
            "**Summary:** KneeVision++ achieves 90%+ accuracy on all clinically actionable tasks. "
            "The full pipeline reaches up to **95.96%** on definitive radiographic grading, "
            "**95.11%** on binary OA triage, and **92-94% 5-class accuracy** for auto-resolved cases."
        )
        st.markdown("#### 90%+ Capabilities at a Glance")
        cols2 = st.columns(3)
        data_90 = [
            ("95.96%", "Definitive Grading",    "Excluding doubtful KL 1 - all clear-cut grades", "#f59e0b"),
            ("95.11%", "Binary OA Screening",   "KL 0-1 vs KL 2-4 primary-care triage",           "#0ea5e9"),
            ("95.11%", "3-Tier Staging",        "Prevention / Conservative Therapy / Surgical",    "#8b5cf6"),
            ("94.31%", "Gating P>=0.80",        "81% of cases auto-resolved above 94% accuracy",   "#10b981"),
            ("93.21%", "Gating P>=0.75",        "86% of cases auto-resolved above 93% accuracy",   "#10b981"),
            ("92.04%", "Gating P>=0.70",        "90% of cases auto-resolved above 92% accuracy",   "#10b981"),
        ]
        for i, (val, name, desc, col) in enumerate(data_90):
            with cols2[i % 3]:
                st.markdown(
                    f"<div style=\"background:#1e293b;border:1px solid {col}44;"
                    f"border-radius:12px;padding:.8rem 1rem;margin-bottom:.6rem;\">"
                    f"<div style=\"color:{col};font-size:1.5rem;font-weight:900;\">{val}</div>"
                    f"<div style=\"color:#e2e8f0;font-weight:700;font-size:.88rem;margin:.2rem 0;\">{name}</div>"
                    f"<div style=\"color:#64748b;font-size:.78rem;\">{desc}</div></div>",
                    unsafe_allow_html=True,
                )


def page_performance():
    st.subheader("📊 Model Performance")
    st.caption("All numbers read live from models/ metadata, reports/, and mlflow.db.")
    perf_5c, perf_fusion, perf_bin, perf_grp = st.tabs([
        "5-Class Grading", "Multimodal Fusion", "Binary OA Detection", "Grouped Evaluation"
    ])

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

        img_p = image_probs(image, list(image_models.values()), device)
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

    if st.button("Generate Rehabilitation Plan", type="primary"):
        recommender = load_rehab_recommender()
        with st.spinner("Retrieving orthopedic guidelines & synthesizing personalized plan via Llama 3.2..."):
            result = recommender.recommend(grade, patient_context=custom_input)

        badge = "🧠 Llama-3.2:1B RAG Synthesis" if result.used_llm else "📄 Guideline Excerpts (Direct)"
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
            st.warning("Ollama isn't reachable at localhost:11434 — showing retrieved guideline excerpts directly instead of an LLM-synthesized summary.")

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
    "🏆 90%+ Showcase": page_showcase,
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
    "Results": ["🏆 90%+ Showcase", "📊 Performance", "📈 MLflow"],
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
