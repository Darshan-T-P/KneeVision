"""Joint-sized crop windows for wide-field knee radiographs.

The classifiers are trained on tight crops where the joint line fills the frame (224x224).
A clinical AP radiograph crop that spans femoral shaft to tibial shaft shrinks the joint
space and osteophytes to a few pixels once resized, and the predicted grade then depends
on crop scale and contrast rather than anatomy. This module cuts square windows around
the joint at several scales, normalises contrast, and reports whether the scales agree.

It does not find the joint automatically (a simple profile detector was tried and was
worse than a fixed height), so the joint height is a parameter the UI exposes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageOps

DEFAULT_JOINT_Y = 0.50
# Window side as a fraction of the half-image (single knee) width. Wide enough to keep the
# femoral condyles and tibial plateau, tight enough that the joint space stays visible.
DEFAULT_SCALES = (0.40, 0.55, 0.70)


def window_box(width: int, height: int, joint_y_frac: float = DEFAULT_JOINT_Y,
               frac: float = 0.55) -> tuple[int, int, int, int]:
    """Square window (x1, y1, x2, y2) centred horizontally and on `joint_y_frac` vertically.

    The window is shifted, never shrunk, to stay inside the image; its side is capped at
    the shorter image dimension.
    """
    side = max(32, min(int(round(frac * width)), width, height))
    cx, cy = width // 2, int(round(joint_y_frac * height))
    x1 = min(max(0, cx - side // 2), width - side)
    y1 = min(max(0, cy - side // 2), height - side)
    return x1, y1, x1 + side, y1 + side


def enhance_contrast(image: Image.Image, cutoff: float = 1.0) -> Image.Image:
    """Stretch the grey-level histogram (clipping `cutoff`% at each end); returns RGB."""
    return ImageOps.autocontrast(image.convert("L"), cutoff=cutoff).convert("RGB")


def knee_windows(half: Image.Image, joint_y_frac: float = DEFAULT_JOINT_Y,
                 scales: Sequence[float] = DEFAULT_SCALES, enhance: bool = True) -> list[Image.Image]:
    """Joint-centred square crops of `half` at each scale, optionally contrast-normalised."""
    out = []
    for frac in scales:
        crop = half.convert("RGB").crop(window_box(half.width, half.height, joint_y_frac, frac))
        out.append(enhance_contrast(crop) if enhance else crop)
    return out


@dataclass
class ScaleReport:
    probs: np.ndarray            # mean class probabilities over scales
    per_scale_pred: list[int]    # argmax at each scale
    spread: int                  # max - min predicted grade across scales
    scale_sensitive: bool        # True when the answer depends on crop scale

    @property
    def pred(self) -> int:
        return int(self.probs.argmax())

    @property
    def confidence(self) -> float:
        return float(self.probs.max())


def multiscale_report(windows: Sequence[Image.Image],
                      predict_probs: Callable[[Image.Image], np.ndarray],
                      max_spread: int = 1, min_confidence: float = 0.7) -> ScaleReport:
    """Average `predict_probs` over the windows and flag unstable results.

    A result is scale-sensitive when per-scale grades differ by more than `max_spread`
    or the averaged confidence is below `min_confidence`. 0.7 comes from 451 labelled test
    crops (CORAL model): predictions under 0.7 were exactly right ~49% of the time vs ~71%
    above it, so the grade should not be presented as a single confident answer.
    """
    per = [np.asarray(predict_probs(w), dtype=float) for w in windows]
    mean = np.mean(per, axis=0)
    preds = [int(p.argmax()) for p in per]
    spread = max(preds) - min(preds)
    return ScaleReport(mean, preds, spread, bool(spread > max_spread or mean.max() < min_confidence))
