from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from PIL import Image

from kneevision.clinical.prepare import RADIOGRAPHIC_FIELDS

# Real per-compartment OARSI grades (0-3) available per knee from the OAI
# clinical CSV (see scripts/download_oai.py). Each is a 4-way classification
# target for the auxiliary heads in KneeXRayClassifier. Kept as one alias of
# the canonical field list in kneevision.clinical.prepare so the image and
# text branches can never drift apart on which fields exist.
AUX_GRADE_FIELDS = RADIOGRAPHIC_FIELDS
AUX_NUM_GRADES = {field: 4 for field in AUX_GRADE_FIELDS}
AUX_IGNORE_INDEX = -100  # matches nn.CrossEntropyLoss default ignore_index


def _grade_label(value) -> int:
    """Parse a grade cell (int, float-string, or missing "") into a
    class index, or AUX_IGNORE_INDEX if missing/unparseable."""
    if value is None or value == "":
        return AUX_IGNORE_INDEX
    try:
        grade = int(float(value))
    except (TypeError, ValueError):
        return AUX_IGNORE_INDEX
    return grade if 0 <= grade <= 3 else AUX_IGNORE_INDEX


class KneeXRayAuxDataset(Dataset):
    """X-ray dataset that also returns real per-compartment OARSI grades as
    auxiliary multi-task labels, matched by patient ID + side (from the OAI
    clinical CSV) the same way `fusion.dataset.MultimodalDataset` matches
    clinical text. Missing grades are marked with AUX_IGNORE_INDEX so the
    per-head CrossEntropyLoss(ignore_index=AUX_IGNORE_INDEX) skips them.
    """

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        features_dict: dict[tuple[str, str], dict[str, Any]] | None = None,
        transform=None,
        minority_transform=None,
        minority_labels=None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.minority_transform = minority_transform
        self.minority_labels = minority_labels or set()

        features_dict = features_dict or {}
        self.aux_labels: list[dict[str, int]] = []
        for path in image_paths:
            stem = path.stem
            matched = {}
            if len(stem) >= 2:
                patient_id = stem[:-1]
                side_char = stem[-1].upper()
                side = "left" if side_char == "L" else "right" if side_char == "R" else ""
                matched = features_dict.get((patient_id, side)) or features_dict.get((patient_id, "")) or {}
            self.aux_labels.append({field: _grade_label(matched.get(field)) for field in AUX_GRADE_FIELDS})

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict[str, int]]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.minority_transform and self.labels[idx] in self.minority_labels:
            img = self.minority_transform(img)
        elif self.transform:
            img = self.transform(img)
        return img, self.labels[idx], self.aux_labels[idx]


def collate_aux_batch(batch):
    """Collate (image, label, aux_dict) samples into batched tensors,
    keeping aux labels as a dict[str, LongTensor]."""
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    aux = {
        field: torch.tensor([b[2][field] for b in batch], dtype=torch.long)
        for field in AUX_GRADE_FIELDS
    }
    return images, labels, aux
