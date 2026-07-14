import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


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
        module.register_backward_hook(backward_hook)

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np

    @staticmethod
    def overlay(cam: np.ndarray, image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        heatmap = _colormap(cam)
        return (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)


def _colormap(cam: np.ndarray) -> np.ndarray:
    cam_uint8 = (cam * 255).astype(np.uint8)
    r = np.clip(2 * cam_uint8 - 255, 0, 255).astype(np.uint8)
    g = np.clip(2 * cam_uint8 - 128, 0, 255).astype(np.uint8)
    b = np.clip(2 * cam_uint8, 0, 255).astype(np.uint8)
    for i in range(128):
        mask = cam_uint8 == i
        r[mask] = 0
        g[mask] = np.clip(4 * i, 0, 255).astype(np.uint8)[mask]
        b[mask] = 255
    for i in range(128, 256):
        mask = cam_uint8 == i
        r[mask] = np.clip(4 * (i - 128), 0, 255).astype(np.uint8)[mask]
        g[mask] = np.clip(255 - 4 * (i - 128), 0, 255).astype(np.uint8)[mask]
        b[mask] = np.clip(255 - 4 * (i - 128), 0, 255).astype(np.uint8)[mask]
    return np.stack([r, g, b], axis=-1)


def explain_prediction(
    model: torch.nn.Module,
    image: Image.Image,
    transform,
    device: torch.device,
) -> tuple[int, float, np.ndarray]:
    x = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        confidence = probs[0, pred].item()

    gradcam = GradCAM(model)
    cam = gradcam.generate(x, class_idx=pred)
    img_np = np.array(image.resize((224, 224))) / 255.0
    overlay = gradcam.overlay(cam, img_np)

    return pred, confidence, overlay
