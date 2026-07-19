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


AVAILABLE_MODELS = {
    "densenet121",
    "efficientnet-b4",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "swin_t",
    "swin_s",
    "swin_b",
}


def _get_backbone_and_dim(model_name: str):
    """Return (backbone, in_features) for CNN-style models."""
    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        backbone = models.densenet121(weights=weights)
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        return backbone, in_features

    if model_name == "efficientnet-b4":
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b4(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        return backbone, in_features

    if model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        backbone = models.convnext_tiny(weights=weights)
        in_features = backbone.classifier[2].in_features
        backbone.classifier = nn.Identity()
        return backbone, in_features

    if model_name == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        backbone = models.convnext_small(weights=weights)
        in_features = backbone.classifier[2].in_features
        backbone.classifier = nn.Identity()
        return backbone, in_features

    if model_name == "convnext_base":
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        backbone = models.convnext_base(weights=weights)
        in_features = backbone.classifier[2].in_features
        backbone.classifier = nn.Identity()
        return backbone, in_features

    raise ValueError(f"Unsupported CNN model: {model_name}")


class KneeXRayClassifier(nn.Module):
    def __init__(self, model_name: str = "densenet121", num_classes: int = 5, ordinal: bool = False):
        super().__init__()
        self.model_name = model_name
        self.ordinal = ordinal
        out_features = num_classes - 1 if ordinal else num_classes

        if model_name in {"densenet121", "efficientnet-b4", "convnext_tiny", "convnext_small", "convnext_base"}:
            backbone, in_features = _get_backbone_and_dim(model_name)
            self.backbone = backbone
            self.classifier = ImprovedHead(in_features, out_features)

        elif model_name == "vit_b_16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            model = models.vit_b_16(weights=weights)
            in_features = model.hidden_dim
            model.heads = nn.Identity()
            self.backbone = model
            self.classifier = ImprovedHead(in_features, out_features)

        elif model_name == "vit_b_32":
            weights = models.ViT_B_32_Weights.IMAGENET1K_V1
            model = models.vit_b_32(weights=weights)
            in_features = model.hidden_dim
            model.heads = nn.Identity()
            self.backbone = model
            self.classifier = ImprovedHead(in_features, out_features)

        elif model_name == "vit_l_16":
            weights = models.ViT_L_16_Weights.IMAGENET1K_V1
            model = models.vit_l_16(weights=weights)
            in_features = model.hidden_dim
            model.heads = nn.Identity()
            self.backbone = model
            self.classifier = ImprovedHead(in_features, out_features)

        elif model_name == "swin_t":
            weights = models.Swin_T_Weights.IMAGENET1K_V1
            model = models.swin_t(weights=weights)
            in_features = model.head.in_features
            model.head = nn.Identity()
            self.backbone = model
            self.classifier = ImprovedHead(in_features, out_features)

        elif model_name == "swin_s":
            weights = models.Swin_S_Weights.IMAGENET1K_V1
            model = models.swin_s(weights=weights)
            in_features = model.head.in_features
            model.head = nn.Identity()
            self.backbone = model
            self.classifier = ImprovedHead(in_features, out_features)

        elif model_name == "swin_b":
            weights = models.Swin_B_Weights.IMAGENET1K_V1
            model = models.swin_b(weights=weights)
            in_features = model.head.in_features
            model.head = nn.Identity()
            self.backbone = model
            self.classifier = ImprovedHead(in_features, out_features)

        else:
            raise ValueError(f"Unsupported model: {model_name}. Choose from {sorted(AVAILABLE_MODELS)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
