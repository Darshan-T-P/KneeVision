
import torch
from PIL import Image

from kneevision.data.aux_dataset import (
    KneeXRayAuxDataset,
    AUX_GRADE_FIELDS,
    AUX_IGNORE_INDEX,
    collate_aux_batch,
    _grade_label,
)


def test_grade_label_parses_valid_and_missing():
    assert _grade_label("2") == 2
    assert _grade_label(3) == 3
    assert _grade_label("") == AUX_IGNORE_INDEX
    assert _grade_label(None) == AUX_IGNORE_INDEX
    assert _grade_label("not_a_number") == AUX_IGNORE_INDEX
    assert _grade_label("9") == AUX_IGNORE_INDEX  # out of the 0-3 OARSI range


def test_aux_dataset_matches_by_patient_id_and_side(tmp_path):
    img_right = tmp_path / "9000100R.png"
    img_left = tmp_path / "9000100L.png"
    img_unmatched = tmp_path / "9999999R.png"
    for p in (img_right, img_left, img_unmatched):
        Image.new("RGB", (32, 32)).save(p)

    features_dict = {
        ("9000100", "right"): {"jsn_m": "2", "osteophyte_m": "1", "sclerosis_m": ""},
        ("9000100", "left"): {"jsn_m": "0", "attrition_l": "3"},
    }
    ds = KneeXRayAuxDataset(
        image_paths=[img_right, img_left, img_unmatched],
        labels=[2, 0, 1],
        features_dict=features_dict,
    )
    assert len(ds) == 3
    assert ds.aux_labels[0]["jsn_m"] == 2
    assert ds.aux_labels[0]["osteophyte_m"] == 1
    assert ds.aux_labels[0]["sclerosis_m"] == AUX_IGNORE_INDEX  # empty cell -> ignored
    assert ds.aux_labels[1]["jsn_m"] == 0
    assert ds.aux_labels[1]["attrition_l"] == 3
    # unmatched patient id: every field falls back to ignore-index
    assert all(v == AUX_IGNORE_INDEX for v in ds.aux_labels[2].values())


def test_aux_dataset_getitem_and_collate(tmp_path):
    img = tmp_path / "9000100R.png"
    Image.new("RGB", (32, 32)).save(img)
    ds = KneeXRayAuxDataset(
        image_paths=[img, img],
        labels=[1, 2],
        features_dict={("9000100", "right"): {"jsn_m": "1"}},
        transform=lambda im: torch.zeros(3, 8, 8),
    )
    batch = [ds[0], ds[1]]
    images, labels, aux = collate_aux_batch(batch)
    assert images.shape == (2, 3, 8, 8)
    assert labels.tolist() == [1, 2]
    assert set(aux.keys()) == set(AUX_GRADE_FIELDS)
    assert aux["jsn_m"].tolist() == [1, 1]
    assert aux["osteophyte_m"].tolist() == [AUX_IGNORE_INDEX, AUX_IGNORE_INDEX]
