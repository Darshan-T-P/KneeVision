import numpy as np

from kneevision.xai.base import _colormap, overlay_heatmap


def test_colormap_scalar():
    heatmap = _colormap(np.array(0.5))
    assert heatmap.shape == (1, 1, 3)
    assert heatmap.dtype == np.uint8


def test_colormap_vector():
    heatmap = _colormap(np.array([0.0, 1.0]))
    assert heatmap.shape == (1, 2, 3)


def test_colormap_matrix():
    cam = np.random.rand(64, 64)
    heatmap = _colormap(cam)
    assert heatmap.shape == (64, 64, 3)
    assert heatmap.dtype == np.uint8


def test_overlay_heatmap_shapes_match():
    cam = np.random.rand(224, 224)
    image = np.random.rand(224, 224, 3)
    overlay = overlay_heatmap(cam, image)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8


def test_overlay_heatmap_resizes():
    cam = np.random.rand(224, 224)
    image = np.random.rand(300, 400, 3)
    overlay = overlay_heatmap(cam, image)
    assert overlay.shape == (300, 400, 3)
