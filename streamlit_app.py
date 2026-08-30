"""KneeVision++ demo — Streamlit app for multimodal knee OA diagnosis.

Run with:  uv run --extra dev streamlit run streamlit_app.py
"""

import os
import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image

from kneevision.utils.helpers import get_device
from kneevision.data.transforms import val_transform
from kneevision.models.image_model import load_trained_model
from kneevision.training.losses import ordinal_to_probs
from kneevision.xai import gradcam_explain, scorecam_explain, lime_explain

MODELS_DIR = Path(__file__).resolve().parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
DEMO_DIR = Path(__file__).resolve().parent / "data" / "raw" / "test"
KL_LABELS = {0: "Normal", 1: "Doubtful", 2: "Mild", 3: "Moderate", 4: "Severe"}

DEMO_REPORTS = {
    0: ("FINDINGS: The medial and lateral femorotibial joint spaces are well preserved. "
        "No osteophyte formation, subchondral sclerosis, or cystic change is identified. "
        "Patellofemoral compartment is normal. Periarticular soft tissues are unremarkable.\n\n"
        "IMPRESSION: No radiographic evidence of knee osteoarthritis."),
    1: ("FINDINGS: Minute marginal osteophyte formation is noted at the medial femoral condyle. "
        "Joint spaces remain preserved. No definite subchondral sclerosis or cyst formation.\n\n"
        "IMPRESSION: Doubtful (minimal) osteoarthritis of the right knee — KL grade 1."),
    2: ("FINDINGS: Small-to-moderate marginal osteophytes arise from the medial tibial plateau and "
        "femoral condyle. Subtle narrowing of the medial compartment is present. Possible early "
        "subchondral sclerosis. Intercondylar eminence is sharp.\n\n"
        "IMPRESSION: Mild medial compartment osteoarthritis — KL grade 2."),
    3: ("FINDINGS: Definite narrowing of the medial femorotibial joint space with moderate multiple "
        "marginal osteophytes. Moderate subchondral sclerosis and small pseudocystic areas with "
        "sclerotic borders. Slight varus alignment of the knee.\n\n"
        "IMPRESSION: Moderate osteoarthritis of the knee — KL grade 3."),
    4: ("FINDINGS: Severe near-complete loss of the medial joint space with a bone-on-bone appearance. "
        "Large osteophytes project from all compartments. Marked subchondral sclerosis with defined "
        "pseudocysts. Gross deformity of the femoral and tibial articular surfaces.\n\n"
        "IMPRESSION: Severe tricompartmental osteoarthritis — KL grade 4."),
}

GRADE_FEATURES = {
    0: "Normal joint — intact joint space, no osteophytes or sclerosis.",
    1: "Doubtful — minute osteophytes, joint space still preserved.",
    2: "Mild — small osteophytes with possible joint space narrowing.",
    3: "Moderate — definite narrowing, moderate osteophytes, sclerosis.",
    4: "Severe — bone-on-bone, large osteophytes, marked deformity.",
}


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
    section.main > div { padding-top: 1.2rem; }

    .kvp-hero {
        background: linear-gradient(120deg, #0f766e 0%, #0e7490 55%, #0369a1 100%);
        border-radius: 18px; padding: 2rem 2.2rem; color: #ffffff;
        margin-bottom: 1.4rem;
    }
    .kvp-hero h1 { color: #ffffff; margin: 0 0 .3rem 0; font-size: 2.1rem; }
    .kvp-hero p  { color: #d8f3f0; margin: 0; font-size: 1.02rem; }
    .kvp-badges span {
        display:inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.35);
        border-radius: 999px; padding: .15rem .7rem; font-size: .78rem; margin-right: .45rem; margin-top: .8rem;
        color: #ecfeff;
    }
    .kvp-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 1.1rem 1.3rem; box-shadow: 0 1px 3px rgba(15,23,42,.06); height: 100%;
    }
    .kvp-card h4 { margin: 0 0 .55rem 0; color: #334155; font-size: .95rem;
                   text-transform: uppercase; letter-spacing: .06em; }
    .kvp-grade { font-size: 2.5rem; font-weight: 800; line-height: 1.05; }
    .kvp-sub   { color: #64748b; font-size: .92rem; margin-top: .15rem; }
    .kvp-bar   { background:#e2e8f0; border-radius:999px; height:.65rem; overflow:hidden; margin-top:.7rem; }
    .kvp-fill  { height:100%; border-radius:999px;
                 background:linear-gradient(90deg,#0ea5e9,#0f766e); }

    .flow-row { display:flex; align-items:center; gap:.55rem; margin:.45rem 0; flex-wrap:wrap; }
    .flow-node {
        background:#f0fdfa; border:1.5px solid #99f6e4; color:#134e4a;
        border-radius:12px; padding:.55rem .95rem; font-size:.88rem; font-weight:600; text-align:center;
    }
    .flow-node small { display:block; font-weight:400; color:#0f766e; font-size:.74rem; }
    .flow-node.fusion { background:#eff6ff; border-color:#bfdbfe; color:#1e3a8a; }
    .flow-node.output { background:#fefce8; border-color:#fde68a; color:#713f12; }
    .flow-arrow { color:#94a3b8; font-size:1.25rem; font-weight:700; }
    .flow-spacer { flex:0 0 3.2rem; }

    .chip-ok   { background:#dcfce7; color:#166534; border-radius:999px; padding:.25rem .85rem;
                 font-weight:700; display:inline-block; }
    .chip-warn { background:#fef9c3; color:#854d0e; border-radius:999px; padding:.25rem .85rem;
                 font-weight:700; display:inline-block; }
</style>
"""


def hero():
    st.markdown(
        """
        <div class="kvp-hero">
            <h1>&#129495; KneeVision++ &mdash; Multimodal OA Diagnosis</h1>
            <p>Explainable deep learning for knee osteoarthritis: KL grading from X-rays,
            screening from clinical reports, and a late-fusion demo combining both.</p>
            <div class="kvp-badges"><span>Kellgren&ndash;Lawrence 0&ndash;4</span>
            <span>CNN Ensemble</span><span>BioClinicalBERT</span><span>Grad-CAM &middot; LIME</span>
            <span>MLflow-tracked</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body_html: str):
    return f'<div class="kvp-card"><h4>{title}</h4>{body_html}</div>'


def grade_card(title: str, pred: int | None, conf: float | None, accent: str) -> str:
    if pred is None:
        return card(title, '<div class="kvp-sub">—</div>')
    pct = f"{conf:.1%}" if conf is not None else "—"
    fill_w = int((conf or 0) * 100)
    return card(
        title,
        f'<div class="kvp-grade" style="color:{accent}">KL {pred}</div>'
        f'<div class="kvp-sub">{KL_LABELS[pred]}</div>'
        f'<div class="kvp-bar"><div class="kvp-fill" style="width:{fill_w}%"></div></div>'
        f'<div class="kvp-sub">confidence {pct}</div>',
    )


def fusion_flow_diagram():
    st.markdown(
        """
        <div class="kvp-card">
          <h4>Multimodal late-fusion architecture</h4>
          <div class="flow-row">
            <div class="flow-node">&#129707; Knee X-ray<small>224&times;224 radiograph</small></div>
            <div class="flow-arrow">&rarr;</div>
            <div class="flow-node">CNN Ensemble<small>DenseNet121 &middot; EfficientNet-B4</small></div>
            <div class="flow-arrow">&rarr;</div>
            <div class="flow-node">P(KL 0&ndash;4)<small>image modality</small></div>
          </div>
          <div class="flow-row">
            <div class="flow-node">&#128221; Clinical report<small>free-text findings</small></div>
            <div class="flow-arrow">&rarr;</div>
            <div class="flow-node">BioClinicalBERT<small>frozen encoder + head</small></div>
            <div class="flow-arrow">&rarr;</div>
            <div class="flow-node">P(KL 0&ndash;4)<small>text modality</small></div>
          </div>
          <div class="flow-row">
            <div class="flow-spacer"></div>
            <div class="flow-arrow">&#8618;</div>
            <div class="flow-node fusion">&#8853; Late Fusion<small>P<sub>fused</sub> = (1&minus;&alpha;)&middot;P<sub>img</sub> + &alpha;&middot;P<sub>text</sub></small></div>
            <div class="flow-arrow">&rarr;</div>
            <div class="flow-node output">Final KL grade<small>argmax + confidence</small></div>
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
        meta = json.loads(path.read_text())
        meta["file"] = path.stem + ".pt"
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
        try:
            available[meta["model_name"]] = load_trained_model(
                path, device, num_classes=meta.get("num_classes", 5)
            )
        except RuntimeError:
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
    try:
        model = load_trained_model(MODELS_DIR / meta["file"], device, num_classes=2)
        return model, meta["model_name"]
    except RuntimeError:
        return None, None


@st.cache_resource(show_spinner="Loading clinical model...")
def load_clinical_model():
    device = get_device()
    from kneevision.clinical.model import ClinicalTextModel

    trained = MODELS_DIR / "best_clinical.pt"
    try:
        model = ClinicalTextModel(num_classes=5).to(device)
        if trained.exists():
            model.load_state_dict(torch.load(trained, map_location=device, weights_only=True))
            return model, True
        return model, False
    except Exception:
        return None, False


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
        st.markdown(card("Dataset", "<b>8,260</b> labeled knee X-rays<br><small>Kaggle MOST-style set · train/val/test split</small>"), unsafe_allow_html=True)
    with col_m:
        st.markdown(card("Clinical corpus", "<b>16,592</b> OAI reports<br><small>narrative radiology findings CSV</small>"), unsafe_allow_html=True)
    with col_r:
        st.markdown(card("Best screening", "<b>87.0%</b> accuracy · AUC 0.946<br><small>ConvNeXt-S binary OA detector</small>"), unsafe_allow_html=True)

    fusion_flow_diagram()

    st.markdown("**What we built**")
    st.markdown(
        """
        1. **Data** — 8,260 Kaggle knee X-rays (train/val/test) + OAI clinical reports CSV.
        2. **5-class KL grading** — fine-tuned DenseNet121 and EfficientNet-B4 with class-weighted loss,
           cosine LR schedule, early stopping on quadratic-weighted κ, EMA checkpointing.
        3. **Dedicated binary OA detection** (KL 0-1 vs 2-4) — same recipe with a 2-class head;
           ConvNeXt-Small joined as the strongest member.
        4. **Class grouping from 5-class probabilities** — marginalizing softmax outputs into
           2 groups (0-1 / 2-4) and 3 groups (0-1 / 2-3 / 4).
        5. **Ensembling + TTA + threshold tuning** on the validation split for all tasks.
        6. **Explainability** — Grad-CAM / Score-CAM / LIME heatmaps in this app.
        7. **Multimodal fusion** — CNN + BioClinicalBERT late fusion (demo in the sidebar menu).
        8. **Tracking & serving** — MLflow experiment tracking, Streamlit demo (this app).
        """
    )

    st.markdown("**Headline test-set results**")
    acc_5c, acc_gb, acc_g3 = report_accuracy("evaluate"), report_accuracy("grouped_binary"), report_accuracy("grouped_3class")
    headline = pd.DataFrame([
        {"Task": "5-class grading", "Approach": "DenseNet121", "Test accuracy": f"{acc_5c:.1%}" if acc_5c else "—", "ROC-AUC": "—", "Cohen's κ": "0.7751 (val)"},
        {"Task": "Binary OA detection", "Approach": "ConvNeXt-Small (best single)", "Test accuracy": "87.0%", "ROC-AUC": "0.9457", "Cohen's κ": "0.7571 (val)"},
        {"Task": "Binary OA detection", "Approach": "DenseNet + ConvNeXt ensemble", "Test accuracy": "86.5%", "ROC-AUC": "0.9463", "Cohen's κ": "—"},
        {"Task": "Grouped (2-group)", "Approach": "Marginalized 5-class probs", "Test accuracy": f"{acc_gb:.1%}" if acc_gb else "84.96%", "ROC-AUC": "0.9302", "Cohen's κ": "—"},
        {"Task": "Grouped (3-group)", "Approach": "Marginalized 5-class probs", "Test accuracy": f"{acc_g3:.1%}" if acc_g3 else "70.17%", "ROC-AUC": "—", "Cohen's κ": "—"},
    ])
    st.table(headline)
    st.caption("κ marked (val) is validation κ at training time; artifacts live in Model Performance. "
               "Published KL-grading systems reach ~65–75% 5-class accuracy — adjacent-grade confusion dominates.")


def page_diagnosis(device, image_models):
    st.subheader("🩻 X-ray Diagnosis")
    st.caption("Upload a knee radiograph — get a KL grade, per-class confidences, and an independent OA screening verdict.")

    col_img, col_ui = st.columns([1.1, 1])
    with col_ui:
        model_choice = st.selectbox("Model", list(image_models.keys()) + ["Ensemble (all)"])
        uploaded = st.file_uploader("Upload a knee X-ray", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded is None:
        with col_img:
            st.info("⬅ Upload an X-ray to see predictions here.")
        return

    image = Image.open(uploaded).convert("RGB")
    with col_img:
        st.image(image, caption="Uploaded X-ray", width="stretch")

    models = [image_models[m] for m in (image_models if model_choice == "Ensemble (all)" else [model_choice])]
    probs = image_probs(image, models, device)
    pred, confidence = int(probs.argmax()), float(probs.max())

    c_grade, c_screen, c_chart = st.columns([1, 1, 1.4])

    with c_grade:
        st.markdown(grade_card(f"KL grade — {model_choice}", pred, confidence, "#0f766e"), unsafe_allow_html=True)
        st.caption("Grades: 0 Normal · 1 Doubtful · 2 Mild · 3 Moderate · 4 Severe")

    bin_model, bin_name = load_binary_model()
    with c_screen:
        if bin_model is not None:
            p_oa = binary_oa_prob(bin_model, image, device)
            verdict = ('<span class="chip-warn">OA detected</span>'
                       if p_oa >= 0.5 else '<span class="chip-ok">No OA</span>')
            body = (f"<div style='font-size:2rem;font-weight:800;color:#0369a1'>{p_oa:.1%}</div>"
                    f"<div class='kvp-sub'>probability of OA (KL ≥ 2)</div>"
                    f"<div class='kvp-bar'><div class='kvp-fill' style='width:{int(p_oa*100)}%'></div></div>"
                    f"<div style='margin-top:.6rem'>{verdict}</div>")
            st.markdown(card(f"OA screening — {bin_name}", body), unsafe_allow_html=True)
            st.caption("Independent dedicated binary detector (KL 0-1 vs 2-4), threshold 0.50.")
        else:
            st.markdown(card("OA screening", "<div class='kvp-sub'>No binary checkpoint found.</div>"), unsafe_allow_html=True)

    with c_chart:
        st.markdown("**Class probabilities**")
        st.bar_chart(pd.Series(probs, index=[f"G{i} {KL_LABELS[i]}" for i in range(5)]), height=250)


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

    if xai_method == "Grad-CAM":
        pred, conf, overlay = gradcam_explain(model, image, val_transform, device)
        importance, segments = None, None
    elif xai_method == "Score-CAM":
        pred, conf, overlay = scorecam_explain(model, image, val_transform, device)
        importance, segments = None, None
    else:
        pred, conf, importance, segments = lime_explain(model, image, val_transform, device)
        overlay = None

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(image, caption="Original", width="stretch")
    with c2:
        shown = overlay if overlay is not None else importance
        kind = f"{xai_method} heatmap" if overlay is not None else "LIME superpixels"
        st.image(shown, caption=f"{kind} — predicted KL {pred}", width="stretch")
    with c3:
        st.markdown(card("Interpretation",
                         f"<div class='kvp-sub' style='font-size:1rem;color:#334155'>Predicted "
                         f"<b>KL Grade {pred}</b> ({KL_LABELS[pred]}).</div>"
                         "<div class='kvp-sub' style='margin-top:.5rem'>Bright regions are the evidence the model "
                         "used — typically the joint space and osteophyte margins.</div>"),
                   unsafe_allow_html=True)


def page_clinical(device):
    st.subheader("📝 Clinical Report Diagnosis")
    st.caption("BioClinicalBERT reads the free-text radiology report and predicts the KL grade — no image needed.")

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
    report = st.text_area(
        "Radiology report text",
        placeholder="e.g. FINDINGS: Moderate joint space narrowing. Multiple moderate osteophytes. "
                    "Moderate subchondral sclerosis. IMPRESSION: Moderate osteoarthritis.",
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


def page_fusion(device, image_models):
    st.subheader("🔀 Multimodal Fusion Demo")
    st.caption("Combine what the CNN sees in the X-ray with what BioClinicalBERT reads in the report.")

    clinical_model, clinical_trained = load_clinical_model()
    fusion_flow_diagram()
    st.write("")

    col_img, col_txt = st.columns(2)
    with col_img:
        up = st.file_uploader("1 · Upload knee X-ray", type=["png", "jpg", "jpeg", "bmp"], key="fusion_img")
    with col_txt:
        report = st.text_area("2 · Paste radiology report",
                              placeholder="FINDINGS: ... IMPRESSION: ...", height=140)

    alpha = st.slider("Fusion weight α (text influence)", 0.0, 1.0, 0.5, 0.05,
                      help="0 = image only, 1 = text only")

    ready_img = up is not None
    ready_txt = bool(report.strip()) and clinical_model is not None
    if st.button("Run fusion", type="primary", disabled=not (ready_img or ready_txt)):
        img_p = txt_p = None
        if ready_img:
            img_p = image_probs(Image.open(up).convert("RGB"), list(image_models.values()), device)
        if ready_txt:
            txt_p = clinical_probs(clinical_model, report, device)

        if img_p is not None and txt_p is not None:
            fused = (1 - alpha) * img_p + alpha * txt_p
        else:
            fused = img_p if img_p is not None else txt_p

        ipred = int(img_p.argmax()) if img_p is not None else None
        tpred = int(txt_p.argmax()) if txt_p is not None else None
        fpred, fconf = int(fused.argmax()), float(fused.max())

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(grade_card("🩻 Image says", ipred, float(img_p[ipred]) if ipred is not None else None, "#0f766e"),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(grade_card("📝 Text says", tpred, float(txt_p[tpred]) if tpred is not None else None, "#7c3aed"),
                        unsafe_allow_html=True)
        with c3:
            st.markdown(grade_card("⊕ Fused decision", fpred, fconf, "#b45309"), unsafe_allow_html=True)

        st.bar_chart(pd.DataFrame({
            "Image": img_p if img_p is not None else np.zeros(5),
            "Text": txt_p if txt_p is not None else np.zeros(5),
            "Fused": fused,
        }, index=[f"G{i}" for i in range(5)]), height=280)

        if not clinical_trained:
            st.caption("Note: the text branch is untrained (random weights) — treat its output as a shape demo only.")
    elif not ready_img and not ready_txt:
        st.info("Provide an X-ray and/or a report to run the fusion.")


def page_performance():
    st.subheader("📊 Model Performance")
    st.caption("All numbers read live from models/ metadata, reports/, and mlflow.db.")
    perf_5c, perf_bin, perf_grp = st.tabs(["5-Class Grading", "Binary OA Detection", "Grouped Evaluation"])

    with perf_5c:
        left, right = st.columns([1, 1])
        with left:
            rows = [
                {"Model": m["model_name"], "Best val κ": round(m["best_kappa"], 4),
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

    with perf_bin:
        left, right = st.columns([1, 1])
        with left:
            rows = [
                {"Model": m["model_name"], "Best val κ": round(m["best_kappa"], 4),
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
    st.caption("Pick a grade to see a real test-set X-ray, its matching radiology report, "
               "and how every model reacts — image CNN, clinical BERT, and the binary OA screener.")

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
        st.markdown("**🩻 X-ray (ground truth)**")
        image = Image.open(available[grade]).convert("RGB")
        st.image(image, caption=f"Test split sample · true label KL {grade}", width="stretch")

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
            expect = True if grade >= 2 else False
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



    st.subheader("📈 MLflow Tracking")
def page_mlflow():
    st.subheader("📈 MLflow Tracking")
    df = mlflow_runs_table()
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
    "🩻 X-ray Diagnosis": lambda: page_diagnosis(*load_image_models()),
    "🔥 Explainability": lambda: page_xai(*load_image_models()),
    "📝 Clinical Text": lambda: page_clinical(load_image_models()[0]),
    "🔀 Multimodal Fusion": lambda: page_fusion(*load_image_models()),
    "🧪 Guided Demo": lambda: page_demo(*load_image_models()),
    "📊 Performance": page_performance,
    "📈 MLflow": page_mlflow,
}


def main():
    st.set_page_config(page_title="KneeVision++", page_icon="🦵", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            "<div style='font-size:1.5rem;font-weight:800;padding:.4rem 0 .1rem 0'>🦵 KneeVision++</div>",
            unsafe_allow_html=True,
        )
        st.caption("Multimodal knee OA diagnosis\n\nKL 0-4 · CNN + BERT · Explainable")
        page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
        st.divider()
        st.caption("Checkpoints: `models/best_*.pt`\n\nTracking: `sqlite:///mlflow.db`")

    device, image_models = load_image_models()
    if not image_models:
        st.error("No loadable image checkpoints found in models/. Train a model first.")
        return
    PAGES[page]()


if __name__ == "__main__":
    main()
