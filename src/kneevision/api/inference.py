"""Shared single-sample inference helpers for the API routes."""
import numpy as np
import torch
from PIL import Image

from kneevision.data.transforms import val_transform
from kneevision.training.losses import ordinal_to_probs


@torch.inference_mode()
def predict_xray_probs(model, image: Image.Image, device) -> np.ndarray:
    model.eval()
    x = val_transform(image).unsqueeze(0).to(device)
    logits = model(x)
    if getattr(model, "ordinal", False):
        probs = ordinal_to_probs(logits)
    else:
        probs = torch.softmax(logits, dim=1)
    return probs.squeeze(0).cpu().numpy()


@torch.inference_mode()
def predict_clinical_probs(model, text: str, device) -> np.ndarray:
    model.eval()
    enc = model._encode([text], device)
    logits = model(enc["input_ids"], enc["attention_mask"])
    if getattr(model, "ordinal", False):
        probs = ordinal_to_probs(logits)
    else:
        probs = torch.softmax(logits, dim=1)
    return probs.squeeze(0).cpu().numpy()
