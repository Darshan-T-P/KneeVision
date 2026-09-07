from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from kneevision.models.image_model import KneeXRayClassifier
from kneevision.clinical.model import ClinicalTextModel
from kneevision.data.transforms import val_transform
from kneevision.training.losses import ordinal_to_probs


class MultimodalFusionModel(nn.Module):
    """Multimodal Late/Joint Fusion model combining X-ray CNN visual representations
    with BioClinicalBERT clinical text representations for knee OA severity grading.
    """
    def __init__(
        self,
        image_model: KneeXRayClassifier,
        text_model: ClinicalTextModel,
        num_classes: int = 5,
        ordinal: bool = True,
        freeze_encoders: bool = True,
        fusion_dim: int = 512,
    ):
        super().__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.num_classes = num_classes
        self.ordinal = ordinal

        if freeze_encoders:
            for param in self.image_model.parameters():
                param.requires_grad_(False)
            for param in self.text_model.parameters():
                param.requires_grad_(False)

        # Dynamically infer feature dimension of image model
        if hasattr(image_model, "classifier") and hasattr(image_model.classifier, "net"):
            img_feat_dim = image_model.classifier.net[1].in_features
        else:
            img_feat_dim = 1024

        txt_feat_dim = text_model.hidden_dim
        out_features = num_classes - 1 if ordinal else num_classes

        self.fusion_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(img_feat_dim + txt_feat_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, out_features),
        )

    def extract_features(
        self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract image features, text features, and fused feature vector."""
        img_feats = self.image_model.extract_features(image)
        if img_feats.dim() > 2:
            img_feats = torch.flatten(img_feats, 1)

        txt_feats = self.text_model.extract_features(input_ids, attention_mask)
        fused_feats = torch.cat([img_feats, txt_feats], dim=1)
        return img_feats, txt_feats, fused_feats

    def forward(
        self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        _, _, fused_feats = self.extract_features(image, input_ids, attention_mask)
        return self.fusion_head(fused_feats)

    def predict_probs(
        self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return calibrated class probability distribution [B, num_classes]."""
        logits = self.forward(image, input_ids, attention_mask)
        if self.ordinal:
            return ordinal_to_probs(logits)
        return F.softmax(logits, dim=-1)

    @torch.inference_mode()
    def predict(
        self, image: Image.Image | torch.Tensor, text: str, device: torch.device
    ) -> tuple[int, float, np.ndarray]:
        """High-level single-sample inference accepting PIL image and clinical text."""
        self.eval()

        if isinstance(image, Image.Image):
            img_t = val_transform(image).unsqueeze(0).to(device)
        elif isinstance(image, torch.Tensor):
            img_t = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
        else:
            raise TypeError("image must be PIL Image or torch.Tensor")

        enc = self.text_model._encode([text], device)
        probs = self.predict_probs(img_t, enc["input_ids"], enc["attention_mask"])
        probs_np = probs.squeeze(0).cpu().numpy()
        pred_class = int(probs_np.argmax())
        confidence = float(probs_np[pred_class])

        return pred_class, confidence, probs_np


def _infer_fusion_ordinal(state_dict: dict, num_classes: int) -> bool:
    for key, value in state_dict.items():
        if "fusion_head" in key and key.endswith(".weight") and ("5.weight" in key or "4.weight" in key):
            return value.shape[0] == num_classes - 1
    return True


def _infer_image_submodel_ordinal(state_dict: dict, num_classes: int) -> bool:
    """The image sub-model's own classifier head may have a different ordinal
    setting than the fusion head (e.g. a non-ordinal image checkpoint fused
    under an ordinal fusion head) — infer it from its own weight shape."""
    key = "image_model.classifier.net.5.weight"
    if key in state_dict:
        return state_dict[key].shape[0] == num_classes - 1
    return False


def _infer_text_submodel_ordinal(state_dict: dict, num_classes: int) -> bool:
    key = "text_model.classifier.weight"
    if key in state_dict:
        return state_dict[key].shape[0] == num_classes - 1
    return False


def load_trained_fusion_model(
    path: str | Path,
    device: torch.device,
    num_classes: int = 5,
    img_model_name: str = "densenet121",
) -> MultimodalFusionModel:
    """Load a trained MultimodalFusionModel from checkpoint with auto-architecture detection."""
    data = torch.load(path, map_location=device, weights_only=False)
    if isinstance(data, dict) and "model_state_dict" in data:
        state = data["model_state_dict"]
        ordinal = bool(data.get("ordinal", _infer_fusion_ordinal(state, num_classes)))
    else:
        state = data
        ordinal = _infer_fusion_ordinal(state, num_classes)

    img_ordinal = _infer_image_submodel_ordinal(state, num_classes)
    txt_ordinal = _infer_text_submodel_ordinal(state, num_classes)

    img_model = KneeXRayClassifier(model_name=img_model_name, num_classes=num_classes, ordinal=img_ordinal).to(device)
    txt_model = ClinicalTextModel(num_classes=num_classes, ordinal=txt_ordinal).to(device)

    model = MultimodalFusionModel(
        image_model=img_model,
        text_model=txt_model,
        num_classes=num_classes,
        ordinal=ordinal,
        freeze_encoders=True,
    ).to(device)

    model.load_state_dict(state)
    model.eval()
    return model
