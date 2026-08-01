import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from kneevision.config.settings import CLINICAL_MODEL_NAME, CLINICAL_MAX_LENGTH


class ClinicalTextModel(nn.Module):
    """BioClinicalBERT encoder with an optional classification head.

    `extract_features()` returns the pooled [CLS] embedding so the same
    module can feed the Phase 5 multimodal fusion model.
    """

    def __init__(self, model_name: str = CLINICAL_MODEL_NAME, num_classes: int = 5,
                 freeze_encoder: bool = False, max_length: int = CLINICAL_MAX_LENGTH):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def _encode(self, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoding.items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))

    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    @torch.inference_mode()
    def predict(self, texts: list[str], device: torch.device) -> tuple[list[int], list[float]]:
        self.eval()
        encoding = self._encode(texts, device)
        logits = self(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).cpu().tolist()
        confidences = probs.max(dim=1).values.cpu().tolist()
        return preds, confidences
