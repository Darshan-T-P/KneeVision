import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


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


def overlay_heatmap(cam: np.ndarray, image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heatmap = _colormap(cam)
    return (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)


def get_prediction(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> tuple[int, float, torch.Tensor]:
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        confidence = probs[0, pred].item()
    return pred, confidence, logits
