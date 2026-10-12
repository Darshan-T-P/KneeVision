"""Unit tests for bilateral radiograph detection and anatomical dual-knee extraction."""

from pathlib import Path
import numpy as np
from PIL import Image

from kneevision.data.bilateral import (
    detect_bilateral_radiograph,
    detect_knee_prosthesis,
    extract_bilateral_knees,
)


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


def test_detect_knee_prosthesis_sample():
    prosthetic_path = Path("data/demo/sample_prosthetic_crop.png")
    native_path = Path("data/demo/sample_native_crop.png")

    if prosthetic_path.exists():
        p_img = Image.open(prosthetic_path)
        is_p, conf_p, details_p = detect_knee_prosthesis(p_img)
        assert is_p is True
        assert conf_p > 0.0
        assert details_p["bright_frac"] >= 0.038

    if native_path.exists():
        n_img = Image.open(native_path)
        is_n, conf_n, _ = detect_knee_prosthesis(n_img)
        assert is_n is False
        assert conf_n == 0.0


def test_detect_knee_prosthesis_synthetic():
    # Synthetic image with bright metallic block (flat saturation)
    arr = np.full((200, 200), 50, dtype=np.uint8)
    arr[60:140, 60:140] = 250  # Dense metal prosthesis block
    img = Image.fromarray(arr)

    is_p, conf, details = detect_knee_prosthesis(img)
    assert is_p is True
    assert conf > 0.0
    assert details["bright_std"] < 24.0

    # Synthetic normal bone image with realistic density distribution
    arr_normal = np.zeros((200, 200), dtype=np.uint8)
    arr_normal[20:180, 20:180] = 80
    arr_normal[90:110, 90:110] = 130
    arr_normal[98:102, 98:102] = 250
    img_normal = Image.fromarray(arr_normal)

    is_norm, conf_norm, _ = detect_knee_prosthesis(img_normal)
    assert is_norm is False
    assert conf_norm == 0.0

