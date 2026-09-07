import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as mcm
from kneevision.training.losses import ordinal_to_class


def _colormap(cam: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim == 0:
        cam = np.array([[cam.item()]])
    elif cam.ndim == 1:
        cam = cam.reshape(1, -1)
    cam = np.clip(cam, 0.0, 1.0)
    return (mcm.jet(cam)[..., :3] * 255).astype(np.uint8)


def overlay_heatmap(cam: np.ndarray, image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a CAM heatmap over the source image. `image` may be either
    0-1 float (the convention used by every caller in this package) or
    already 0-255 — scaled up to 0-255 automatically so the blend weights
    apply to matching ranges (mixing a 0-255 heatmap with a 0-1 image made
    the heatmap swamp the original image almost completely)."""
    heatmap = _colormap(cam)
    if heatmap.shape[:2] != image.shape[:2]:
        from PIL import Image as PILImage
        heatmap = np.array(PILImage.fromarray(heatmap).resize((image.shape[1], image.shape[0])))
    image_255 = image * 255.0 if image.max() <= 1.0 else image.astype(np.float32)
    return np.clip(alpha * heatmap + (1 - alpha) * image_255, 0, 255).astype(np.uint8)


def get_prediction(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> tuple[int, float, torch.Tensor]:
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        if getattr(model, "ordinal", False):
            pred = ordinal_to_class(logits).item()
            confidence = 1.0
        else:
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            confidence = probs[0, pred].item()
    return pred, confidence, logits
