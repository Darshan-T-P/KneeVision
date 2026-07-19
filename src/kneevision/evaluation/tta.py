import torch
import torch.nn.functional as F
from PIL import Image
from kneevision.data.transforms import tta_transforms_list
from kneevision.training.losses import ordinal_to_class


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
        if getattr(model, "ordinal", False):
            pred = ordinal_to_class(logits)
            p = torch.zeros(5, device=device)
            p[pred] = 1.0
            probs += p
        else:
            probs += F.softmax(logits, dim=1).squeeze()

    probs /= len(tta_transforms_list)
    pred = probs.argmax().item()
    confidence = probs[pred].item()
    return pred, confidence
