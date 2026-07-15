import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from .base import get_prediction


def _segment_grid(h: int, w: int, grid_size: int = 7) -> np.ndarray:
    seg = np.zeros((h, w), dtype=np.int32)
    cell_h, cell_w = h // grid_size, w // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            y0, y1 = i * cell_h, (i + 1) * cell_h if i < grid_size - 1 else h
            x0, x1 = j * cell_w, (j + 1) * cell_w if j < grid_size - 1 else w
            seg[y0:y1, x0:x1] = i * grid_size + j
    return seg


def explain(
    model: torch.nn.Module,
    image: Image.Image,
    transform,
    device: torch.device,
    grid_size: int = 7,
    num_samples: int = 500,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    x = transform(image).unsqueeze(0).to(device)
    pred, confidence, _ = get_prediction(model, x, device)

    img_np = np.array(image.resize((224, 224))) / 255.0
    h, w = 224, 224

    segments = _segment_grid(h, w, grid_size)
    n_segments = segments.max() + 1

    perturbations = np.random.binomial(1, 0.5, size=(num_samples, n_segments)).astype(np.float32)
    perturbations[0] = 1.0
    distances = np.sqrt(n_segments - perturbations.sum(axis=1))

    mean_color = np.mean(img_np, axis=(0, 1))
    scores = np.zeros(num_samples)

    for i in range(num_samples):
        masked = img_np.copy()
        for seg_id in range(n_segments):
            if perturbations[i, seg_id] == 0:
                mask = segments == seg_id
                masked[mask] = mean_color
        tensor = transform(Image.fromarray((masked * 255).astype(np.uint8))).unsqueeze(0).to(device)
        with torch.inference_mode():
            scores[i] = model(tensor)[0, pred].item()

    kernel_width = 0.25
    weights = np.exp(-(distances**2) / (kernel_width**2))

    scores_norm = scores - scores.min()
    scores_norm = scores_norm / (scores_norm.max() + 1e-8)

    X = perturbations
    y = scores_norm
    X_reg = X.T * weights
    reg = np.eye(n_segments) * 1e-6
    theta = np.linalg.lstsq(X_reg @ X + reg, X_reg @ y, rcond=None)[0]

    importance_map = np.zeros((h, w), dtype=np.float32)
    for seg_id in range(n_segments):
        importance_map[segments == seg_id] = theta[seg_id]

    importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)

    return pred, confidence, importance_map, segments.astype(np.float32)
