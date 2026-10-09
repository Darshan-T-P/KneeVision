import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from kneevision.training.losses import ordinal_to_class, ordinal_to_probs

from .base import get_prediction, overlay_heatmap


class ScoreCAM:
    def __init__(self, model: torch.nn.Module, target_layer: str = "backbone.features.denseblock4"):
        self.model = model
        self.model.eval()
        self.activations = None
        self._register_hook(target_layer)

    def _register_hook(self, target_layer: str):
        module = dict(self.model.named_modules())[target_layer]

        def forward_hook(_, __, output):
            self.activations = output

        module.register_forward_hook(forward_hook)

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        _ = self.model(x)

        if class_idx is None:
            with torch.inference_mode():
                logits = self.model(x)
                if getattr(self.model, "ordinal", False):
                    class_idx = ordinal_to_class(logits).item()
                else:
                    class_idx = logits.argmax(dim=1).item()

        _B, C, _H, _W = self.activations.shape
        activations = self.activations.detach()

        weights = torch.zeros(C, device=x.device)
        upsampled = F.interpolate(activations, size=x.shape[2:], mode="bilinear", align_corners=False)

        for i in range(C):
            cam_i = upsampled[0, i].unsqueeze(0).unsqueeze(0)
            cam_i = (cam_i - cam_i.min()) / (cam_i.max() - cam_i.min() + 1e-8)
            masked = x * cam_i
            with torch.inference_mode():
                logits = self.model(masked)
                if getattr(self.model, "ordinal", False):
                    # Was a hard 0/1 indicator (masked prediction == target
                    # grade, exactly), which is far noisier than the
                    # continuous softmax score used on the non-ordinal
                    # branch below -- most masked channels would score
                    # exactly 0 even when they partially support the target
                    # grade. Use the predicted grade's own probability mass
                    # (ordinal_to_probs) for a continuous, comparable score.
                    probs = ordinal_to_probs(logits)
                    target_idx = min(class_idx, probs.shape[1] - 1)
                    score = probs[0, target_idx]
                else:
                    score = F.softmax(logits, dim=1)[0, class_idx]
            weights[i] = score

        weights = weights.view(1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().detach().cpu().numpy()
        if cam_np.ndim == 0:
            cam_np = np.array([[cam_np.item()]])
        elif cam_np.ndim == 1:
            cam_np = cam_np.reshape(1, -1)
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np


def explain(
    model: torch.nn.Module,
    image: Image.Image,
    transform,
    device: torch.device,
) -> tuple[int, float, np.ndarray]:
    x = transform(image).unsqueeze(0).to(device)
    pred, confidence, _ = get_prediction(model, x, device)

    scorecam = ScoreCAM(model)
    cam = scorecam.generate(x, class_idx=pred)
    img_np = np.array(image.resize((224, 224))) / 255.0
    overlay = overlay_heatmap(cam, img_np)

    return pred, confidence, overlay
