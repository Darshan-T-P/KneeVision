"""KneeVision++ demo — Streamlit app for X-ray diagnosis, XAI, and clinical text.

Run with:  uv run --extra dev streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

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
KL_LABELS = {0: "Normal", 1: "Doubtful", 2: "Mild", 3: "Moderate", 4: "Severe"}

st.set_page_config(page_title="KneeVision++", page_icon="🦵", layout="wide")
st.title("🦵 KneeVision++")
st.caption("Explainable AI for knee osteoarthritis (OA) severity grading — KL grade 0-4")


@st.cache_resource(show_spinner="Loading models...")
def load_image_models():
    device = get_device()
    available = {}
    for name in ["densenet121", "vit_b_16"]:
        for candidate in [f"best_{name}.pt", f"best_{name}_ordinal.pt"]:
            path = MODELS_DIR / candidate
            if not path.exists():
                continue
            try:
                available[name] = load_trained_model(path, device)
                break
            except RuntimeError:
                continue
    return device, available


@st.cache_resource(show_spinner="Loading clinical model...")
def load_clinical_model():
    device = get_device()
    from kneevision.clinical.model import ClinicalTextModel

    trained = MODELS_DIR / "best_clinical.pt"
    model = ClinicalTextModel(num_classes=5).to(device)
    if trained.exists():
        model.load_state_dict(torch.load(trained, map_location=device, weights_only=True))
        return model, True
    return model, False


@torch.inference_mode()
def predict_image(model, image: Image.Image, device) -> tuple[int, float, np.ndarray]:
    model.eval()
    x = val_transform(image).unsqueeze(0).to(device)
    logits = model(x)
    if getattr(model, "ordinal", False):
        probs = ordinal_to_probs(logits).squeeze(0)
        pred = int(probs.argmax().item())
        confidence = float(probs[pred].item())
    else:
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred = int(probs.argmax().item())
        confidence = float(probs[pred].item())
    return pred, confidence, probs.cpu().numpy()


def main():
    device, image_models = load_image_models()
    if not image_models:
        st.error("No loadable image checkpoints found in models/. Train a model first.")
        return

    tab_pred, tab_xai, tab_clinical, tab_perf = st.tabs(
        ["🔬 X-ray Diagnosis", "🔍 Explainability", "📝 Clinical Text", "📊 Model Performance"]
    )

    # ---------------- X-ray Diagnosis ----------------
    with tab_pred:
        st.subheader("Knee X-ray Diagnosis")
        col_img, col_ui = st.columns([1, 1])
        with col_ui:
            model_choice = st.selectbox("Model", list(image_models.keys()) + ["Ensemble (all)"])
            uploaded = st.file_uploader("Upload a knee X-ray", type=["png", "jpg", "jpeg", "bmp"])

        if uploaded is None:
            with col_img:
                st.info("Upload a knee X-ray image to get a KL grade prediction.")
            return

        image = Image.open(uploaded).convert("RGB")
        models = [image_models[m] for m in (image_models if model_choice == "Ensemble (all)" else [model_choice])]

        probs = np.zeros(5)
        for model in models:
            pred, conf, p = predict_image(model, image, device)
            probs += p
        probs /= len(models)
        pred = int(probs.argmax())
        confidence = float(probs[pred])

        with col_img:
            st.image(image, caption="Uploaded X-ray", width="stretch")

        col_res, col_plot = st.columns([1, 1])
        with col_res:
            st.markdown(f"### Predicted: **KL Grade {pred}** — {KL_LABELS[pred]}")
            st.progress(confidence, text=f"Confidence: {confidence:.1%}")
            st.caption("KL grades: 0 Normal · 1 Doubtful · 2 Mild · 3 Moderate · 4 Severe")
        with col_plot:
            st.bar_chart(
                pd.Series(probs, index=[f"Grade {i}\n{KL_LABELS[i]}" for i in range(5)]),
                height=300,
            )

    # ---------------- Explainability ----------------
    with tab_xai:
        st.subheader("Explainability — what drives the prediction?")
        st.caption("Grad-CAM / Score-CAM / LIME highlight the image regions the model focuses on (DenseNet model).")
        xai_uploaded = st.file_uploader("Upload a knee X-ray for explanation", type=["png", "jpg", "jpeg", "bmp"],
                                        key="xai_upload")
        xai_method = st.radio("Explanation method", ["Grad-CAM", "Score-CAM", "LIME"], horizontal=True)

        if xai_uploaded is None:
            st.info("Upload an X-ray to see the explanation heatmaps.")
        else:
            image = Image.open(xai_uploaded).convert("RGB")
            model = image_models.get("densenet121") or next(iter(image_models.values()))

            if xai_method == "Grad-CAM":
                pred, conf, overlay = gradcam_explain(model, image, val_transform, device)
            elif xai_method == "Score-CAM":
                pred, conf, overlay = scorecam_explain(model, image, val_transform, device)
            else:
                pred, conf, importance, segments = lime_explain(model, image, val_transform, device)
                overlay = None

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(image, caption="Original", width="stretch")
            if overlay is not None:
                with c2:
                    st.image(overlay, caption=f"{xai_method} overlay (KL {pred})", width="stretch")
            else:
                with c2:
                    st.image(importance, caption=f"LIME importance (KL {pred})", width="stretch")
            with c3:
                st.markdown("### Interpretation")
                st.write(f"Predicted **KL Grade {pred}** ({KL_LABELS[pred]}).")
                st.write("Bright regions = where the model looks to make its decision.")

    # ---------------- Clinical Text ----------------
    with tab_clinical:
        st.subheader("Clinical Text Prediction")
        st.caption("BioClinicalBERT predicts KL grade from a radiology report.")
        clinical_model, is_trained = load_clinical_model()
        if not is_trained:
            st.warning("⚠️  No trained clinical checkpoint found (models/best_clinical.pt). "
                       "Showing a random-initialized demo — train with scripts/train_clinical.py first.")
        report = st.text_area(
            "Radiology report text",
            placeholder="e.g. FINDINGS: Moderate joint space narrowing. Multiple moderate osteophytes. "
                        "Moderate subchondral sclerosis. IMPRESSION: Moderate osteoarthritis.",
            height=180,
        )
        if st.button("Predict KL grade", type="primary"):
            if not report.strip():
                st.warning("Enter a report first.")
            else:
                preds, confs = clinical_model.predict([report], device)
                pred, confidence = preds[0], confs[0]
                st.success(f"Predicted **KL Grade {pred}** — {KL_LABELS[pred]} (confidence {confidence:.1%})")

    # ---------------- Model Performance ----------------
    with tab_perf:
        st.subheader("Model Performance on Test Set (1,656 X-rays)")
        data = {
            "Model": ["DenseNet121 (ordinal)", "ViT-B/16 (ordinal)", "Ensemble + TTA"],
            "Accuracy": ["64.0%", "64.3%", "62.7%"],
            "Cohen's Kappa": ["0.798", "0.811", "0.790"],
            "Parameters": ["8.0M", "86.6M", "—"],
        }
        st.table(data)
        st.markdown(
            "**Note:** Published 5-class KL grading systems reach ~65-75% accuracy — these results are competitive. "
            "Adjacent-grade confusion (e.g. 1↔2) is the main error source, which is normal for KL grading."
        )
        st.subheader("Dataset")
        dist = {"KL 0": 639, "KL 1": 296, "KL 2": 447, "KL 3": 223, "KL 4": 51}
        st.bar_chart(pd.Series(dist), height=250)


if __name__ == "__main__":
    main()
