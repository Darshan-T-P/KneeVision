import torch
import torch.nn as nn
from torchvision import models


class KneeXRayClassifier(nn.Module):
    def __init__(self, model_name: str = "densenet121", num_classes: int = 5):
        super().__init__()
        if model_name == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            backbone = models.densenet121(weights=weights)
            in_features = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
        elif model_name == "efficientnet-b4":
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
            backbone = models.efficientnet_b4(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
