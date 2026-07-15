import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from kneevision.data.transforms import tta_transforms_list


@torch.inference_mode()
def tta_predict(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
) -> tuple[int, float]:
    model.eval()
    probs = torch.zeros(5, device=device)

    for tta_tfm in tta_transforms_list:
        x = tta_tfm(image).unsqueeze(0).to(device)
        logits = model(x)
        probs += F.softmax(logits, dim=1).squeeze()

    probs /= len(tta_transforms_list)
    pred = probs.argmax().item()
    confidence = probs[pred].item()
    return pred, confidence
