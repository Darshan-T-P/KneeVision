"""Optional radiograph preprocessing (opt-in; the default pipeline applies none).

`paper` approximates the preprocessing of Azad & Anik (IEEE Access 2025): denoising,
sharpening, histogram equalization and contrast enhancement. It uses only PIL so no new
dependency is needed. The same function must be applied at train AND test time.
"""
from PIL import Image, ImageFilter, ImageOps

PREPROCESS_CHOICES = ("none", "hist_eq", "paper")


def _to_rgb(gray: Image.Image) -> Image.Image:
    return gray.convert("RGB")


def hist_eq(img: Image.Image) -> Image.Image:
    """Global histogram equalization on the grayscale radiograph."""
    return _to_rgb(ImageOps.equalize(img.convert("L")))


def paper_pipeline(img: Image.Image) -> Image.Image:
    """Median denoise -> unsharp-mask sharpen -> histogram equalization -> autocontrast."""
    gray = img.convert("L").filter(ImageFilter.MedianFilter(size=3))
    gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    gray = ImageOps.equalize(gray)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    return _to_rgb(gray)


def get_preprocess(name: str):
    """Return a PIL->PIL callable, or None for 'none'."""
    if name not in PREPROCESS_CHOICES:
        raise ValueError(f"unknown preprocess {name!r}; choose from {PREPROCESS_CHOICES}")
    return {"none": None, "hist_eq": hist_eq, "paper": paper_pipeline}[name]
