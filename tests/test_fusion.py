import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from kneevision.fusion.model import (
    MultimodalFusionModel,
    _infer_fusion_ordinal,
    _infer_image_submodel_ordinal,
    _infer_text_submodel_ordinal,
)
from kneevision.fusion.dataset import MultimodalDataset


class DummyBackbone(nn.Module):
    def forward(self, x):
        return torch.randn(x.shape[0], 1024)


class DummyImageModel(nn.Module):
    def __init__(self, num_classes=5, ordinal=True):
        super().__init__()
        self.backbone = DummyBackbone()
        out_dim = num_classes - 1 if ordinal else num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim),
        )

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.classifier(self.extract_features(x))


class DummyTextModel(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=5, ordinal=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ordinal = ordinal
        out_dim = num_classes - 1 if ordinal else num_classes
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def extract_features(self, input_ids, attention_mask):
        return torch.randn(input_ids.shape[0], self.hidden_dim)

    def forward(self, input_ids, attention_mask):
        return self.classifier(self.extract_features(input_ids, attention_mask))

    def _encode(self, texts, device):
        return {
            "input_ids": torch.randint(0, 1000, (len(texts), 32), device=device),
            "attention_mask": torch.ones((len(texts), 32), device=device),
        }


@pytest.fixture
def dummy_fusion_model():
    img_m = DummyImageModel(num_classes=5, ordinal=True)
    txt_m = DummyTextModel(hidden_dim=768, num_classes=5, ordinal=True)
    model = MultimodalFusionModel(
        image_model=img_m,
        text_model=txt_m,
        num_classes=5,
        ordinal=True,
        freeze_encoders=True,
    )
    return model


def test_fusion_model_forward_shape(dummy_fusion_model):
    batch_size = 4
    imgs = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (batch_size, 32))
    attention_mask = torch.ones(batch_size, 32)

    logits = dummy_fusion_model(imgs, input_ids, attention_mask)
    # Ordinal with 5 classes produces 4 logits
    assert logits.shape == (batch_size, 4)


def test_fusion_model_extract_features(dummy_fusion_model):
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (batch_size, 32))
    attention_mask = torch.ones(batch_size, 32)

    img_f, txt_f, fused_f = dummy_fusion_model.extract_features(imgs, input_ids, attention_mask)
    assert img_f.shape == (batch_size, 1024)
    assert txt_f.shape == (batch_size, 768)
    assert fused_f.shape == (batch_size, 1024 + 768)


def test_fusion_model_predict_probs(dummy_fusion_model):
    batch_size = 3
    imgs = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (batch_size, 32))
    attention_mask = torch.ones(batch_size, 32)

    probs = dummy_fusion_model.predict_probs(imgs, input_ids, attention_mask)
    assert probs.shape == (batch_size, 5)
    # Probs must sum to ~1.0
    np.testing.assert_allclose(probs.sum(dim=1).detach().numpy(), np.ones(batch_size), atol=1e-4)


def test_fusion_model_predict_pil(dummy_fusion_model):
    device = torch.device("cpu")
    dummy_img = Image.new("RGB", (224, 224), color="gray")
    report = "FINDINGS: Moderate narrowing. IMPRESSION: KL 3."

    pred_class, conf, probs = dummy_fusion_model.predict(dummy_img, report, device)
    assert isinstance(pred_class, int)
    assert 0 <= pred_class <= 4
    assert 0.0 <= conf <= 1.0
    assert len(probs) == 5
    assert np.isclose(probs.sum(), 1.0, atol=1e-4)


def test_infer_fusion_ordinal():
    state_ordinal = {"fusion_head.5.weight": torch.randn(4, 512)}
    assert _infer_fusion_ordinal(state_ordinal, num_classes=5) is True

    state_non_ordinal = {"fusion_head.5.weight": torch.randn(5, 512)}
    assert _infer_fusion_ordinal(state_non_ordinal, num_classes=5) is False


def test_infer_submodel_ordinal_independent_of_fusion_head():
    # Regression test: a real checkpoint can have a non-ordinal image branch
    # (5-way classifier) fused under an ordinal fusion head (4-way) and an
    # ordinal text branch (4-way) — each sub-model's ordinality must be
    # inferred from its own weights, not inherited from the fusion head.
    state = {
        "image_model.classifier.net.5.weight": torch.randn(5, 1024),
        "text_model.classifier.weight": torch.randn(4, 768),
        "fusion_head.5.weight": torch.randn(4, 512),
    }
    assert _infer_fusion_ordinal(state, num_classes=5) is True
    assert _infer_image_submodel_ordinal(state, num_classes=5) is False
    assert _infer_text_submodel_ordinal(state, num_classes=5) is True


def test_infer_submodel_ordinal_missing_keys_default_false():
    assert _infer_image_submodel_ordinal({}, num_classes=5) is False
    assert _infer_text_submodel_ordinal({}, num_classes=5) is False


class DummyTokenizer:
    def __call__(self, text, padding=True, truncation=True, max_length=256, return_tensors="pt"):
        return {
            "input_ids": torch.randint(0, 1000, (1, max_length)),
            "attention_mask": torch.ones((1, max_length)),
        }


def test_multimodal_dataset_fallback(tmp_path):
    img1 = tmp_path / "9000100L.png"
    img2 = tmp_path / "9000200R.png"
    Image.new("RGB", (100, 100)).save(img1)
    Image.new("RGB", (100, 100)).save(img2)

    ds = MultimodalDataset(
        image_paths=[img1, img2],
        labels=[0, 3],
        features_dict={},
        allow_fallback_text=True,
        tokenizer=DummyTokenizer(),
    )
    assert len(ds) == 2
    item = ds[0]
    assert "image" in item
    assert "input_ids" in item
    assert "attention_mask" in item
    assert item["label"] == 0
    assert item["input_ids"].shape[0] == ds.max_length
