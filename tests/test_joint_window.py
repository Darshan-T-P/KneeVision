import numpy as np
from PIL import Image

from kneevision.data.bilateral import extract_bilateral_knees, split_bilateral_halves
from kneevision.data.joint_window import enhance_contrast, knee_windows, multiscale_report, window_box


def test_window_is_square_and_inside_image():
    for w, h in [(1406, 2000), (300, 300), (500, 200)]:
        for jy in (0.05, 0.5, 0.95):
            x1, y1, x2, y2 = window_box(w, h, jy, 0.7)
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
            assert x2 - x1 == y2 - y1


def test_window_is_centred_on_joint_height_when_room():
    x1, y1, x2, y2 = window_box(1000, 2000, 0.5, 0.4)
    assert abs((y1 + y2) / 2 - 1000) <= 1 and abs((x1 + x2) / 2 - 500) <= 1


def test_windows_one_per_scale_and_sizes_increase():
    half = Image.new("L", (1000, 1800), 100)
    sizes = [w.width for w in knee_windows(half, scales=(0.4, 0.55, 0.7))]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3 and all(w.mode == "RGB" for w in knee_windows(half))


def test_enhance_contrast_stretches_range():
    arr = np.random.default_rng(0).integers(90, 120, (64, 64)).astype(np.uint8)
    out = np.asarray(enhance_contrast(Image.fromarray(arr)).convert("L"))
    assert out.max() - out.min() > 200


def probs(*v):
    return np.array(v, dtype=float)


def test_agreeing_confident_scales_are_stable():
    seq = iter([probs(0, 0, .1, .8, .1)] * 3)
    rep = multiscale_report([None] * 3, lambda _: next(seq))
    assert rep.pred == 3 and rep.spread == 0 and not rep.scale_sensitive


def test_disagreeing_scales_are_flagged():
    seq = iter([probs(0, 0, .1, .1, .8), probs(0, 0, .8, .1, .1), probs(0, 0, .8, .1, .1)])
    rep = multiscale_report([None] * 3, lambda _: next(seq))
    assert rep.spread == 2 and rep.scale_sensitive


def test_low_confidence_is_flagged_even_when_scales_agree():
    seq = iter([probs(.1, .1, .5, .2, .1)] * 3)
    assert multiscale_report([None] * 3, lambda _: next(seq)).scale_sensitive


def test_bilateral_crops_are_joint_sized_not_whole_knee():
    img = Image.fromarray((np.random.default_rng(1).random((1600, 2000)) * 255).astype(np.uint8))
    rk, lk, _ = extract_bilateral_knees(img, split_ratio=0.5)
    assert rk.width == rk.height and rk.width <= 0.6 * 1000 + 2   # ~0.55 x half-width
    rh, lh = split_bilateral_halves(img, split_ratio=0.5)
    assert rh.width == 1000 and lh.width == 1000
