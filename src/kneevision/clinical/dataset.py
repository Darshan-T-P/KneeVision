from torch.utils.data import Dataset
from transformers import AutoTokenizer

from kneevision.config.settings import CLINICAL_MODEL_NAME, CLINICAL_MAX_LENGTH


class ClinicalTextDataset(Dataset):
    """Pairs of (report_text, kl_grade) tokenized by a clinical tokenizer."""

    def __init__(self, texts: list[str], labels: list[int] | None = None,
                 features: list[dict] | None = None, augment: bool = False,
                 model_name: str = CLINICAL_MODEL_NAME, max_length: int = CLINICAL_MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.features = features
        self.augment = augment
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        if self.features is not None and self.augment:
            from kneevision.clinical.prepare import compose_clinical_report
            text = compose_clinical_report(self.features[idx], augment=self.augment)
        else:
            text = self.texts[idx]
            
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item
