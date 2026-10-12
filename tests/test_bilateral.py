"""Unit tests for bilateral radiograph detection and anatomical dual-knee extraction."""

from pathlib import Path
import numpy as np
from PIL import Image

from kneevision.data.bilateral import detect_bilateral_radiograph, extract_bilateral_knees


def test_detect_bilateral_sample():
    sample_path = Path("data/demo/bilateral_standing_ap.png")
    if not sample_path.exists():
        return

    img = Image.open(sample_path)
    is_bilat, conf, details = detect_bilateral_radiograph(img)

    assert is_bilat is True
    assert conf > 0.5
    assert details["aspect_ratio"] > 1.10
    assert 0.40 <= details["split_ratio"] <= 0.60


def test_detect_single_knee_rejection():
    # Synthetic square single knee crop
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[50:180, 50:180] = 180  # Central bone mass
    img = Image.fromarray(arr)

    is_bilat, conf, details = detect_bilateral_radiograph(img)
    assert is_bilat is False
    assert details["aspect_ratio"] == 1.0


def test_extract_bilateral_knees():
    sample_path = Path("data/demo/bilateral_standing_ap.png")
    if not sample_path.exists():
        return

    img = Image.open(sample_path)
    rk, lk, annotated = extract_bilateral_knees(img, split_ratio=0.5, joint_focus=True)

    assert rk.width > 0 and rk.height > 0
    assert lk.width > 0 and lk.height > 0
    assert annotated.size == img.size
