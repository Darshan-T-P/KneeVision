"""KneeVision++ REST API (Phase 7b) — wraps the existing image / clinical-text /
fusion / RAG rehab inference in HTTP endpoints for a decoupled frontend.

Run with:
    uv run uvicorn kneevision.api.main:app --reload --port 8000
"""
import io

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from kneevision.api import deps
from kneevision.api.inference import predict_clinical_probs, predict_xray_probs
from kneevision.api.schemas import (
    ClinicalRequest,
    FusionResponse,
    GuidelineExcerpt,
    ModelsStatusResponse,
    PredictionResponse,
    RehabRequest,
    RehabResponse,
    probs_to_response,
)
from kneevision.data.transforms import val_transform
from kneevision.utils.helpers import get_device
from kneevision.xai import gradcam_explain, lime_explain, scorecam_explain

app = FastAPI(title="KneeVision++ API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_image(file: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models", response_model=ModelsStatusResponse)
def models_status(
    clinical_model=Depends(deps.get_clinical_model),
    fusion_model=Depends(deps.get_fusion_model),
):
    return ModelsStatusResponse(
        image_models=list(deps.get_available_image_backbones()),
        clinical_available=clinical_model is not None,
        fusion_available=fusion_model is not None,
    )


@app.post("/predict/xray", response_model=PredictionResponse)
def predict_xray(file: UploadFile = File(...), model=Depends(deps.get_image_model)):
    if model is None:
        raise HTTPException(status_code=503, detail="No image checkpoint found (models/best_densenet121.pt)")
    image = _read_image(file)
    probs = predict_xray_probs(model, image, get_device())
    return probs_to_response(probs, int(probs.argmax()))


@app.post("/predict/xray/explain")
def predict_xray_explain(file: UploadFile = File(...), method: str = Form("gradcam"), model=Depends(deps.get_image_model)):
    if model is None:
        raise HTTPException(status_code=503, detail="No image checkpoint found (models/best_densenet121.pt)")
    image = _read_image(file)
    device = get_device()

    if method == "gradcam":
        pred, conf, overlay = gradcam_explain(model, image, val_transform, device)
    elif method == "scorecam":
        pred, conf, overlay = scorecam_explain(model, image, val_transform, device)
    elif method == "lime":
        from kneevision.xai.base import overlay_heatmap
        import numpy as np
        pred, conf, importance, _segments = lime_explain(model, image, val_transform, device)
        img_np = np.array(image.resize((224, 224))) / 255.0
        overlay = overlay_heatmap(importance, img_np)
    else:
        raise HTTPException(status_code=400, detail="method must be one of: gradcam, scorecam, lime")

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="image/png",
        headers={"X-KL-Grade": str(pred), "X-Confidence": f"{conf:.4f}"},
    )


@app.post("/predict/clinical", response_model=PredictionResponse)
def predict_clinical(request: ClinicalRequest, model=Depends(deps.get_clinical_model)):
    if model is None:
        raise HTTPException(status_code=503, detail="No clinical checkpoint found (models/best_clinical.pt)")
    probs = predict_clinical_probs(model, request.report, get_device())
    return probs_to_response(probs, int(probs.argmax()))


@app.post("/predict/fusion", response_model=FusionResponse)
def predict_fusion(
    file: UploadFile = File(...),
    report: str = Form(""),
    img_model=Depends(deps.get_image_model),
    clinical_model=Depends(deps.get_clinical_model),
    fusion_model=Depends(deps.get_fusion_model),
):
    device = get_device()
    image = _read_image(file)

    image_resp = clinical_resp = None
    if img_model is not None:
        img_probs = predict_xray_probs(img_model, image, device)
        image_resp = probs_to_response(img_probs, int(img_probs.argmax()))

    if report.strip() and clinical_model is not None:
        txt_probs = predict_clinical_probs(clinical_model, report, device)
        clinical_resp = probs_to_response(txt_probs, int(txt_probs.argmax()))

    if fusion_model is not None and report.strip():
        pred, _conf, probs = fusion_model.predict(image, report, device)
        fusion_resp = probs_to_response(probs, pred)
    elif image_resp is not None:
        fusion_resp = image_resp  # not enough inputs for real fusion — fall back to image-only
    elif clinical_resp is not None:
        fusion_resp = clinical_resp
    else:
        raise HTTPException(status_code=503, detail="No models available to produce a prediction")

    return FusionResponse(image=image_resp, clinical=clinical_resp, fusion=fusion_resp)


@app.post("/rehab/recommend", response_model=RehabResponse)
def rehab_recommend(request: RehabRequest, recommender=Depends(deps.get_rehab_recommender)):
    if not 0 <= request.kl_grade <= 4:
        raise HTTPException(status_code=400, detail="kl_grade must be 0-4")
    result = recommender.recommend(request.kl_grade, patient_context=request.context)
    return RehabResponse(
        kl_grade=result.kl_grade,
        label={0: "Normal", 1: "Doubtful", 2: "Mild", 3: "Moderate", 4: "Severe"}[result.kl_grade],
        synthesis=result.synthesis,
        used_llm=result.used_llm,
        excerpts=[GuidelineExcerpt(heading=c.heading, source=c.source_path, text=c.text) for c in result.retrieved_chunks],
        disclaimer=result.disclaimer,
    )
