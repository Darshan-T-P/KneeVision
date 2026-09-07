"""API route tests. Uses FastAPI dependency overrides with lightweight fake
models instead of real checkpoints, so these stay fast and don't require
models/*.pt to exist (real end-to-end behavior was verified manually via
curl + screenshots during development)."""
import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image

from kneevision.api import deps
from kneevision.api.main import app
from kneevision.rag import GuidelineChunk, GuidelineRetriever, OllamaClient, OllamaUnavailableError, RehabRecommender


class FakeImageModel(nn.Module):
    """Ignores the input and returns fixed logits favoring KL2."""
    ordinal = False

    def forward(self, x):
        batch = x.shape[0]
        return torch.tensor([[0.5, 1.0, 3.0, 0.5, 0.2]]).repeat(batch, 1)


class FakeClinicalModel:
    ordinal = False

    def eval(self):
        return self

    def _encode(self, texts, device):
        return {"input_ids": torch.zeros(1, 4, dtype=torch.long), "attention_mask": torch.ones(1, 4)}

    def __call__(self, input_ids, attention_mask):
        return torch.tensor([[0.2, 0.2, 0.2, 3.0, 0.1]])


class FakeFusionModel:
    def predict(self, image, text, device):
        probs = np.array([0.05, 0.05, 0.1, 0.7, 0.1])
        return 3, 0.7, probs


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def _sample_image_bytes():
    buf = __import__("io").BytesIO()
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_models_status_reflects_availability(client, tmp_path, monkeypatch):
    (tmp_path / "best_densenet121.pt").touch()
    monkeypatch.setattr(deps, "MODELS_DIR", tmp_path)
    deps.get_available_image_backbones.cache_clear()

    app.dependency_overrides[deps.get_clinical_model] = lambda: None
    app.dependency_overrides[deps.get_fusion_model] = lambda: FakeFusionModel()

    data = client.get("/models").json()
    assert data["image_models"] == ["densenet121"]
    assert data["clinical_available"] is False
    assert data["fusion_available"] is True
    deps.get_available_image_backbones.cache_clear()


def test_predict_xray_returns_kl_grade(client):
    app.dependency_overrides[deps.get_image_model] = lambda: FakeImageModel()
    resp = client.post("/predict/xray", files={"file": ("x.png", _sample_image_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["kl_grade"] == 2  # FakeImageModel favors index 2
    assert body["label"] == "Mild"
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3


def test_predict_xray_503_when_no_checkpoint(client):
    app.dependency_overrides[deps.get_image_model] = lambda: None
    resp = client.post("/predict/xray", files={"file": ("x.png", _sample_image_bytes(), "image/png")})
    assert resp.status_code == 503


def test_predict_xray_400_on_bad_image(client):
    app.dependency_overrides[deps.get_image_model] = lambda: FakeImageModel()
    resp = client.post("/predict/xray", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert resp.status_code == 400


def test_predict_clinical_returns_kl_grade(client):
    app.dependency_overrides[deps.get_clinical_model] = lambda: FakeClinicalModel()
    resp = client.post("/predict/clinical", json={"report": "some findings"})
    assert resp.status_code == 200
    assert resp.json()["kl_grade"] == 3  # FakeClinicalModel favors index 3


def test_predict_clinical_503_when_no_checkpoint(client):
    app.dependency_overrides[deps.get_clinical_model] = lambda: None
    resp = client.post("/predict/clinical", json={"report": "some findings"})
    assert resp.status_code == 503


def test_predict_fusion_uses_fusion_model_when_report_given(client):
    app.dependency_overrides[deps.get_image_model] = lambda: FakeImageModel()
    app.dependency_overrides[deps.get_clinical_model] = lambda: FakeClinicalModel()
    app.dependency_overrides[deps.get_fusion_model] = lambda: FakeFusionModel()

    resp = client.post(
        "/predict/fusion",
        files={"file": ("x.png", _sample_image_bytes(), "image/png")},
        data={"report": "some findings"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["image"]["kl_grade"] == 2
    assert body["clinical"]["kl_grade"] == 3
    assert body["fusion"]["kl_grade"] == 3  # from FakeFusionModel, not just image/clinical


def test_predict_fusion_falls_back_to_image_only_without_report(client):
    app.dependency_overrides[deps.get_image_model] = lambda: FakeImageModel()
    app.dependency_overrides[deps.get_clinical_model] = lambda: FakeClinicalModel()
    app.dependency_overrides[deps.get_fusion_model] = lambda: FakeFusionModel()

    resp = client.post("/predict/fusion", files={"file": ("x.png", _sample_image_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["clinical"] is None  # no report -> clinical branch skipped
    assert body["fusion"]["kl_grade"] == body["image"]["kl_grade"]  # fell back to image-only


def test_predict_fusion_503_when_nothing_available(client):
    app.dependency_overrides[deps.get_image_model] = lambda: None
    app.dependency_overrides[deps.get_clinical_model] = lambda: None
    app.dependency_overrides[deps.get_fusion_model] = lambda: None
    resp = client.post("/predict/fusion", files={"file": ("x.png", _sample_image_bytes(), "image/png")})
    assert resp.status_code == 503


def _fake_recommender():
    chunk = GuidelineChunk(text="Exercise helps.", source_path="test.md", kl_grade="2", topic="test", heading="Exercise")

    class StubLLM(OllamaClient):
        def generate(self, prompt):
            raise OllamaUnavailableError("no server in tests")

    return RehabRecommender(GuidelineRetriever([chunk]), llm=StubLLM())


def test_rehab_recommend(client):
    app.dependency_overrides[deps.get_rehab_recommender] = _fake_recommender
    resp = client.post("/rehab/recommend", json={"kl_grade": 2, "context": "mild pain"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["kl_grade"] == 2
    assert body["used_llm"] is False
    assert "not medical advice" in body["disclaimer"].lower()
    assert len(body["excerpts"]) > 0


def test_rehab_recommend_rejects_invalid_kl_grade(client):
    app.dependency_overrides[deps.get_rehab_recommender] = _fake_recommender
    resp = client.post("/rehab/recommend", json={"kl_grade": 9})
    assert resp.status_code == 400


def test_explain_400_on_bad_method(client):
    app.dependency_overrides[deps.get_image_model] = lambda: FakeImageModel()
    resp = client.post(
        "/predict/xray/explain",
        files={"file": ("x.png", _sample_image_bytes(), "image/png")},
        data={"method": "not-a-real-method"},
    )
    assert resp.status_code == 400


def test_explain_503_when_no_checkpoint(client):
    app.dependency_overrides[deps.get_image_model] = lambda: None
    resp = client.post("/predict/xray/explain", files={"file": ("x.png", _sample_image_bytes(), "image/png")})
    assert resp.status_code == 503
