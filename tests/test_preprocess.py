import numpy as np
import pytest
from PIL import Image

from kneevision.data.preprocess import PREPROCESS_CHOICES, get_preprocess
from kneevision.data.transforms import build_train_transform, build_val_transform


def _img():
    rng = np.random.default_rng(0)
    return Image.fromarray((rng.random((64, 48)) * 120 + 40).astype("uint8")).convert("RGB")


def test_none_is_identity_default():
    assert get_preprocess("none") is None


@pytest.mark.parametrize("name", ["hist_eq", "paper"])
def test_preprocess_keeps_size_and_rgb_and_is_deterministic(name):
    fn = get_preprocess(name)
    a, b = fn(_img()), fn(_img())
    assert a.size == (48, 64) and a.mode == "RGB"
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(_img()))


def test_hist_eq_expands_contrast():
    out = np.asarray(get_preprocess("hist_eq")(_img()))[..., 0]
    assert out.max() - out.min() > 200


def test_unknown_name_rejected():
    with pytest.raises(ValueError):
        get_preprocess("clahe")


@pytest.mark.parametrize("name", PREPROCESS_CHOICES)
def test_transforms_accept_preprocess(name):
    assert build_val_transform(preprocess=name)(_img()).shape == (3, 224, 224)
    assert build_train_transform(preprocess=name)(_img()).shape == (3, 224, 224)


def test_robust_transforms_shapes_and_per_image_standardization():
    import torch

    v = build_val_transform(robust=True)(_img())
    assert v.shape == (3, 224, 224) and abs(v.mean().item()) < 1e-4 and abs(v.std().item() - 1) < 1e-3
    for _ in range(5):
        assert build_train_transform(robust=True)(_img()).shape == (3, 224, 224)
    assert torch.isfinite(build_train_transform(robust=True)(_img())).all()
