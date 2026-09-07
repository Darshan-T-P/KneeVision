from pydantic import BaseModel

KL_LABELS = {0: "Normal", 1: "Doubtful", 2: "Mild", 3: "Moderate", 4: "Severe"}


class PredictionResponse(BaseModel):
    kl_grade: int
    label: str
    confidence: float
    probabilities: dict[str, float]


class ClinicalRequest(BaseModel):
    report: str


class FusionResponse(BaseModel):
    image: PredictionResponse | None = None
    clinical: PredictionResponse | None = None
    fusion: PredictionResponse


class RehabRequest(BaseModel):
    kl_grade: int
    context: str = ""


class GuidelineExcerpt(BaseModel):
    heading: str
    source: str
    text: str


class RehabResponse(BaseModel):
    kl_grade: int
    label: str
    synthesis: str
    used_llm: bool
    excerpts: list[GuidelineExcerpt]
    disclaimer: str


class ModelsStatusResponse(BaseModel):
    image_models: list[str]
    clinical_available: bool
    fusion_available: bool


def probs_to_response(probs, kl_grade: int) -> PredictionResponse:
    return PredictionResponse(
        kl_grade=kl_grade,
        label=KL_LABELS[kl_grade],
        confidence=float(probs[kl_grade]),
        probabilities={f"KL{i}": float(probs[i]) for i in range(len(probs))},
    )
