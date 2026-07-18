import torch
import torch.nn as nn
from torchvision import models

_RADIMAGENET_URLS = {
    "densenet121": "https://github.com/BMEII-AI/RadImageNet/releases/download/v1.0/densenet121_radimagenet.pth",
    "efficientnet-b4": "https://github.com/BMEII-AI/RadImageNet/releases/download/v1.0/efficientnet_b4_radimagenet.pth",
}

_RADIMAGENET_IN_FEATURES = {
    "densenet121": 1024,
    "efficientnet-b4": 1792,
}


class KneeXRayClassifier(nn.Module):
    def __init__(self, model_name: str = "densenet121", num_classes: int = 5,
                 ordinal: bool = False, pretrained: str = "radimagenet"):
        super().__init__()
        self.ordinal = ordinal
        out_features = num_classes - 1 if ordinal else num_classes

        backbone = self._build_backbone(model_name)
        in_features = self._get_in_features(model_name, backbone)

        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, out_features),
        )

        if pretrained and pretrained not in ("imagenet", "none"):
            self._load_pretrained(model_name, pretrained)

    @staticmethod
    def _build_backbone(model_name: str) -> nn.Module:
        if model_name == "densenet121":
            backbone = models.densenet121(weights=None)
            backbone.classifier = nn.Identity()
        elif model_name == "efficientnet-b4":
            backbone = models.efficientnet_b4(weights=None)
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        return backbone

    @staticmethod
    def _get_in_features(model_name: str, backbone: nn.Module) -> int:
        try:
            return _RADIMAGENET_IN_FEATURES[model_name]
        except KeyError:
            raise ValueError(f"Unknown in_features for {model_name}")

    def _load_pretrained(self, model_name: str, source: str):
        if source == "radimagenet":
            url = _RADIMAGENET_URLS.get(model_name)
            if url is None:
                print(f"RadImageNet weights not available for {model_name}, using random init")
                return
            print(f"Loading RadImageNet weights for {model_name}...")
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu",
                                                            check_hash=False, progress=True)
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys (classifier): {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
            print("  RadImageNet weights loaded successfully")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
