"""Bilateral Knee Radiograph Detection and Anatomical Joint Extraction.

Handles clinical AP (anteroposterior) standing radiographs containing both knees
in a single image (bilateral views), splitting them into anatomical Right (R)
and Left (L) knee crops for per-joint Kellgren-Lawrence grading.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from kneevision.data.joint_window import DEFAULT_JOINT_Y, window_box


def detect_bilateral_radiograph(image: Image.Image) -> tuple[bool, float, dict[str, Any]]:
    """Detect if an input radiograph contains both knee joints (bilateral AP view).

    Uses geometric aspect ratio and horizontal bone density profile analysis.

    Returns:
        is_bilateral: True if image is likely a bilateral radiograph.
        confidence: Confidence score in [0.0, 1.0].
        details: Dict containing detected split column, aspect ratio, banner bounds, etc.
    """
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size
    aspect = w / h

    gray = np.array(img_rgb.convert("L"), dtype=np.float32)

    # 1. Detect bottom annotation/metadata banner (e.g., patient name, clinic info)
    bottom_cut = h
    bottom_band = gray[int(h * 0.82):, :]
    if bottom_band.size > 0:
        row_means = bottom_band.mean(axis=1)
        # Check for very dark background banner with text
        if row_means[-1] < 35 and np.any(row_means > 40):
            banner_rows = np.where(row_means < 30)[0]
            if len(banner_rows) > 0:
                bottom_cut = int(h * 0.82) + int(banner_rows[0])

    # 2. Analyze horizontal brightness profile in central 40%-60%
    mid_y1 = int(h * 0.20)
    mid_y2 = int(bottom_cut * 0.80)
    mid_band = gray[mid_y1:mid_y2, :]

    col_means = mid_band.mean(axis=0)

    center_start = int(w * 0.35)
    center_end = int(w * 0.65)
    if center_end > center_start:
        valley_idx = int(np.argmin(col_means[center_start:center_end]))
        split_col = center_start + valley_idx
        center_min_val = float(col_means[split_col])
        max_val = float(np.max(col_means)) + 1e-6
        valley_ratio = center_min_val / max_val
    else:
        split_col = w // 2
        valley_ratio = 1.0

    # Bilateral heuristics:
    # 1. Wide aspect ratio: bilateral AP standing views typically have aspect ratio >= 1.10
    # 2. Central valley dip between legs: valley_ratio is low (< 0.75) due to air/soft tissue gap
    is_wide = aspect >= 1.10
    has_valley = valley_ratio <= 0.75

    # Confidence calculation
    conf = 0.0
    if is_wide:
        conf += 0.45 * min(1.0, (aspect - 1.0) / 0.4)
    if has_valley:
        conf += 0.55 * (1.0 - valley_ratio)

    is_bilateral = bool(is_wide and has_valley and conf >= 0.45)

    details = {
        "aspect_ratio": round(aspect, 3),
        "split_col": split_col,
        "split_ratio": round(split_col / w, 3),
        "valley_ratio": round(valley_ratio, 3),
        "bottom_cut": bottom_cut,
        "confidence": round(min(1.0, conf), 3),
    }

    return is_bilateral, min(1.0, conf), details


def extract_bilateral_knees(
    image: Image.Image,
    split_ratio: float | None = None,
    joint_focus: bool = True,
    margin_trim: float = 0.04,
    joint_y_frac: float = DEFAULT_JOINT_Y,
    window_frac: float = 0.55,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Extract Right (patient R = image left) and Left (patient L = image right) knee crops.

    Args:
        image: Original radiograph PIL Image.
        split_ratio: Fractional split position in [0.3, 0.7]. If None, auto-detected.
        joint_focus: If True, returns a joint-sized square window (window_frac x half-width) centred
            at joint_y_frac of the crop height, instead of the whole knee.
        margin_trim: Fraction of top margin to trim (to avoid R/L markers interfering).

    Returns:
        right_knee: PIL Image of patient's Right Knee (R).
        left_knee: PIL Image of patient's Left Knee (L).
        annotated_image: Full image with marked bounding boxes and R / L labels.
    """
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size

    _, _, details = detect_bilateral_radiograph(img_rgb)
    bottom_cut = details.get("bottom_cut", h)

    if split_ratio is None:
        split_col = details.get("split_col", w // 2)
    else:
        split_col = int(w * split_ratio)

    split_col = max(int(w * 0.3), min(int(w * 0.7), split_col))

    top_cut = int(h * margin_trim)

    # Initial crops
    right_raw = img_rgb.crop((0, top_cut, split_col, bottom_cut))
    left_raw = img_rgb.crop((split_col, top_cut, w, bottom_cut))

    rw, rh = right_raw.size
    lw, lh = left_raw.size
    if joint_focus:
        # Joint-sized square window (not the whole knee): matches the tight crops the
        # models were trained on. See kneevision.data.joint_window.
        rb = window_box(rw, rh, joint_y_frac, window_frac)
        lb = window_box(lw, lh, joint_y_frac, window_frac)
        right_knee, left_knee = right_raw.crop(rb), left_raw.crop(lb)
        r_box = (rb[0], top_cut + rb[1], rb[2], top_cut + rb[3])
        l_box = (split_col + lb[0], top_cut + lb[1], split_col + lb[2], top_cut + lb[3])
    else:
        right_knee = right_raw
        left_knee = left_raw
        r_box = (0, top_cut, split_col, bottom_cut)
        l_box = (split_col, top_cut, w, bottom_cut)

    # Create annotated overview image
    annotated = img_rgb.copy()
    draw = ImageDraw.Draw(annotated)

    # Draw boxes
    color_r = (56, 189, 248)  # Cyan
    color_l = (168, 85, 247)  # Purple
    line_w = max(2, int(w * 0.004))

    draw.rectangle(r_box, outline=color_r, width=line_w)
    draw.rectangle(l_box, outline=color_l, width=line_w)

    # Split line
    draw.line([(split_col, 0), (split_col, h)], fill=(234, 179, 8, 180), width=max(1, line_w - 1))

    # Add text labels on the annotated image
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((r_box[0] + 15, r_box[1] + 15), "RIGHT KNEE (R)", fill=color_r, font=font)
    draw.text((l_box[0] + 15, l_box[1] + 15), "LEFT KNEE (L)", fill=color_l, font=font)

    return right_knee, left_knee, annotated


def detect_knee_prosthesis(image: Image.Image) -> tuple[bool, float, dict[str, Any]]:
    """Detect presence of metallic total knee arthroplasty (TKA / TKR prosthesis).

    Metallic implants exhibit hyper-dense radiopacity with sharp boundaries and flat
    internal texture, which fall out-of-distribution (OOD) for native Kellgren-Lawrence models.

    Returns:
        is_prosthesis: True if a metallic knee replacement is detected.
        confidence: Confidence score in [0.0, 1.0].
        details: Dict of diagnostic metrics (bright_frac, bright_std, contrast_ratio).
    """
    arr = np.array(image.convert("L"), dtype=np.float32)
    h, w = arr.shape

    # Focus on central joint compartment
    roi = arr[int(h * 0.18):int(h * 0.78), int(w * 0.18):int(w * 0.82)]
    if roi.size == 0:
        return False, 0.0, {}

    p95 = float(np.percentile(roi, 95))
    p50 = float(np.median(roi))
    contrast_ratio = (p95 - p50) / (p50 + 1e-5)

    roi_min = float(roi.min())
    roi_max = float(roi.max())
    roi_norm = (roi - roi_min) / (roi_max - roi_min + 1e-5) * 255.0

    bright_mask = roi_norm > 185.0
    bright_frac = float(bright_mask.mean())

    if bright_mask.sum() > 100:
        bright_std = float(roi_norm[bright_mask].std())
    else:
        bright_std = 999.0

    # Prosthetic components have high bright plateau with low internal variance
    is_prosthesis = bool(bright_frac >= 0.038 and bright_std < 24.0 and contrast_ratio > 0.40)
    conf = min(1.0, bright_frac / 0.12) if is_prosthesis else 0.0

    details = {
        "bright_frac": round(bright_frac, 4),
        "bright_std": round(bright_std, 1),
        "contrast_ratio": round(float(contrast_ratio), 2),
    }

    return is_prosthesis, round(float(conf), 2), details



def split_bilateral_halves(
    image: Image.Image, split_ratio: float | None = None, margin_trim: float = 0.04,
) -> tuple[Image.Image, Image.Image]:
    """Right (image left) and Left (image right) whole-knee halves, banner and top margin removed.

    Use with `kneevision.data.joint_window.knee_windows` for multi-scale grading.
    """
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size
    _, _, details = detect_bilateral_radiograph(img_rgb)
    split_col = details.get("split_col", w // 2) if split_ratio is None else int(w * split_ratio)
    split_col = max(int(w * 0.3), min(int(w * 0.7), split_col))
    top_cut, bottom_cut = int(h * margin_trim), details.get("bottom_cut", h)
    return (img_rgb.crop((0, top_cut, split_col, bottom_cut)),
            img_rgb.crop((split_col, top_cut, w, bottom_cut)))
