from pathlib import Path
from typing import Any
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from kneevision.config.settings import CLINICAL_MODEL_NAME, CLINICAL_MAX_LENGTH
from kneevision.clinical.prepare import compose_clinical_report, generate_report


class MultimodalDataset(Dataset):
    """Paired multimodal dataset returning Image tensor, tokenized Clinical Text, and KL label."""

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        features_dict: dict[tuple[str, str], dict[str, Any]] | None = None,
        augment_text: bool = False,
        transform=None,
        model_name: str = CLINICAL_MODEL_NAME,
        max_length: int = CLINICAL_MAX_LENGTH,
        allow_fallback_text: bool = False,
        tokenizer: Any = None,
    ):
        self.image_paths: list[Path] = []
        self.labels: list[int] = []
        self.features_list: list[dict[str, Any] | None] = []
        self.raw_texts: list[str | None] = []

        features_dict = features_dict or {}

        for path, label in zip(image_paths, labels):
            stem = path.stem  # e.g. '9001695L' or '9001695R'
            matched_feature = None

            if len(stem) >= 2:
                patient_id = stem[:-1]
                side_char = stem[-1].upper()
                side = "left" if side_char == "L" else "right" if side_char == "R" else ""
                key = (patient_id, side)

                if key in features_dict:
                    matched_feature = features_dict[key]
                elif (patient_id, "") in features_dict:
                    matched_feature = features_dict[(patient_id, "")]

            if matched_feature is not None:
                self.image_paths.append(path)
                self.labels.append(label)
                self.features_list.append(matched_feature)
                self.raw_texts.append(None)
            elif allow_fallback_text:
                self.image_paths.append(path)
                self.labels.append(label)
                self.features_list.append(None)
                self.raw_texts.append(generate_report(label, seed=hash(path.name) % 100000))

        self.transform = transform
        self.augment_text = augment_text
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.features_list[idx] is not None:
            text = compose_clinical_report(self.features_list[idx], augment=self.augment_text)
        elif self.raw_texts[idx] is not None:
            text = self.raw_texts[idx]
        else:
            text = generate_report(self.labels[idx])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "image": img,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": self.labels[idx],
        }
