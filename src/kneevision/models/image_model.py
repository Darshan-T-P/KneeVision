from pathlib import Path
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torchvision import models


class ImprovedHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def __getitem__(self, idx):
        return self.net[idx]

    def __len__(self):
        return len(self.net)


@dataclass(frozen=True)
class BackboneSpec:
    """Describes how to build and strip the head of a torchvision backbone.

    `in_features_attr` is used for backbones that expose the feature dim as an
    attribute (e.g. ViT's `hidden_dim`); otherwise the head is a Linear layer,
    optionally nested at `head_attr[head_index]`.
    """
    factory: Any
    weights: Any
    head_attr: str
    head_index: int | None = None
    in_features_attr: str | None = None

    def build(self) -> tuple[nn.Module, int]:
        backbone = self.factory(weights=self.weights)
        if self.in_features_attr is not None:
            in_features = getattr(backbone, self.in_features_attr)
        else:
            head = getattr(backbone, self.head_attr)
            layer = head[self.head_index] if self.head_index is not None else head
            in_features = layer.in_features
        setattr(backbone, self.head_attr, nn.Identity())
        return backbone, in_features


BACKBONE_REGISTRY: dict[str, BackboneSpec] = {
    "densenet121": BackboneSpec(models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1, "classifier"),
    "efficientnet-b4": BackboneSpec(models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1, "classifier", 1),
    "convnext_tiny": BackboneSpec(models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, "classifier", 2),
    "convnext_small": BackboneSpec(models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1, "classifier", 2),
    "convnext_base": BackboneSpec(models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1, "classifier", 2),
    "vit_b_16": BackboneSpec(models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1, "heads", in_features_attr="hidden_dim"),
    "vit_b_32": BackboneSpec(models.vit_b_32, models.ViT_B_32_Weights.IMAGENET1K_V1, "heads", in_features_attr="hidden_dim"),
    "vit_l_16": BackboneSpec(models.vit_l_16, models.ViT_L_16_Weights.IMAGENET1K_V1, "heads", in_features_attr="hidden_dim"),
    "swin_t": BackboneSpec(models.swin_t, models.Swin_T_Weights.IMAGENET1K_V1, "head"),
    "swin_s": BackboneSpec(models.swin_s, models.Swin_S_Weights.IMAGENET1K_V1, "head"),
    "swin_b": BackboneSpec(models.swin_b, models.Swin_B_Weights.IMAGENET1K_V1, "head"),
}

AVAILABLE_MODELS = set(BACKBONE_REGISTRY)


class KneeXRayClassifier(nn.Module):
    def __init__(self, model_name: str = "densenet121", num_classes: int = 5, ordinal: bool = False):
        super().__init__()
        if model_name not in BACKBONE_REGISTRY:
            raise ValueError(f"Unsupported model: {model_name}. Choose from {sorted(AVAILABLE_MODELS)}")

        self.model_name = model_name
        self.ordinal = ordinal
        out_features = num_classes - 1 if ordinal else num_classes

        backbone, in_features = BACKBONE_REGISTRY[model_name].build()
        self.backbone = backbone
        self.classifier = ImprovedHead(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _infer_ordinal(state_dict: dict, num_classes: int) -> bool:
    """Ordinal heads output num_classes - 1 logits (CORAL)."""
    for key, value in state_dict.items():
        if key.endswith("classifier.net.5.weight"):
            return value.shape[0] == num_classes - 1
    return False


def _infer_model_name(filename: str) -> str:
    stem = Path(filename).stem
    for name in sorted(AVAILABLE_MODELS, key=len, reverse=True):
        if name in stem:
            return name
    return "densenet121"


def load_trained_model(path, device, num_classes: int = 5) -> KneeXRayClassifier:
    """Load a `best_*.pt` raw state dict or a full `checkpoint_*.pt` checkpoint."""
    data = torch.load(path, map_location=device, weights_only=False)

    if isinstance(data, dict) and "model_state_dict" in data:
        state = data["model_state_dict"]
        model_name = data.get("model_name") or _infer_model_name(str(path))
        ordinal = bool(data.get("ordinal", _infer_ordinal(state, num_classes)))
    else:
        state = data
        model_name = _infer_model_name(str(path))
        ordinal = _infer_ordinal(state, num_classes)

    model = KneeXRayClassifier(model_name, num_classes, ordinal=ordinal).to(device)
    model.load_state_dict(state)
    return model
