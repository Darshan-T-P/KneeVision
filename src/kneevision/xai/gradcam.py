import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from kneevision.training.losses import ordinal_to_class, ordinal_to_probs

from .base import get_prediction, overlay_heatmap


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: str = "backbone.features.denseblock4"):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self._register_hooks(target_layer)

    def _register_hooks(self, target_layer: str):
        module = dict(self.model.named_modules())[target_layer]

        def forward_hook(_, __, output):
            self.activations = output

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0]

        module.register_forward_hook(forward_hook)
        module.register_full_backward_hook(backward_hook)

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        x = x.requires_grad_(True)
        logits = self.model(x)
        is_ordinal = getattr(self.model, "ordinal", False)
        if class_idx is None:
            class_idx = ordinal_to_class(logits).item() if is_ordinal else logits.argmax(dim=1).item()

        self.model.zero_grad()
        if is_ordinal:
            # CORAL emits K-1 cumulative "P(grade > k)" logits, not one
            # logit per grade. Backpropping a single raw threshold logit
            # (the old behavior, via a naive index clamp) explains "is this
            # worse than grade k", not "is this grade `class_idx`" -- a
            # semantic mismatch between the heatmap and its own caption.
            # Target the predicted grade's own probability mass instead.
            probs = ordinal_to_probs(logits)
            target_idx = min(class_idx, probs.shape[1] - 1)
            target = probs[0, target_idx]
        else:
            target_idx = min(class_idx, logits.shape[1] - 1)
            target = logits[0, target_idx]
        target.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
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

    gradcam = GradCAM(model)
    cam = gradcam.generate(x, class_idx=pred)
    img_np = np.array(image.resize((224, 224))) / 255.0
    overlay = overlay_heatmap(cam, img_np)

    return pred, confidence, overlay
