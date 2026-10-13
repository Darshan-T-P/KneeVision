import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

from kneevision.config.settings import IMAGE_SIZE
from kneevision.data.preprocess import get_preprocess

randaugment = transforms.RandAugment(num_ops=2, magnitude=9)


def _train_augmentations(size: int):
    return [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop((size, size), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        randaugment,
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]


def _minority_augmentations(size: int):
    return [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop((size, size), scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        randaugment,
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]


def _normalize():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class RandomStretch:
    """Randomly change the aspect ratio (w/h in ~[0.5, 2]) before cropping so the net is not tied to one image shape."""

    def __init__(self, base: int = 256, lo: float = 0.7, hi: float = 1.4):
        self.base, self.lo, self.hi = base, lo, hi

    def __call__(self, img):
        s = random.uniform(self.lo, self.hi)
        return img.resize((int(self.base * s), int(self.base / s)), Image.BICUBIC)


class RandomResolution:
    """Down- then up-sample (simulates low-resolution / different-detector images)."""

    def __init__(self, p: float = 0.5, lo: float = 0.35):
        self.p, self.lo = p, lo

    def __call__(self, img):
        if random.random() > self.p:
            return img
        f = random.uniform(self.lo, 1.0)
        w, h = img.size
        return img.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BILINEAR).resize((w, h), Image.BICUBIC)


class RandomGamma:
    def __init__(self, p: float = 0.7, lo: float = 0.6, hi: float = 1.6):
        self.p, self.lo, self.hi = p, lo, hi

    def __call__(self, img):
        return TF.adjust_gamma(img, random.uniform(self.lo, self.hi)) if random.random() < self.p else img


class AddNoise:
    def __init__(self, max_sigma: float = 0.03):
        self.max_sigma = max_sigma

    def __call__(self, x):
        return x + torch.randn_like(x) * random.uniform(0.0, self.max_sigma)


class PerImageStandardize:
    """Zero-mean / unit-variance per image: removes scanner-specific global brightness and contrast."""

    def __call__(self, x):
        return (x - x.mean()) / (x.std() + 1e-6)


def _robust_augmentations(size: int):
    return [
        RandomStretch(),
        transforms.RandomResizedCrop((size, size), scale=(0.6, 1.0), ratio=(0.6, 1.7)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        RandomGamma(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        randaugment,
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        RandomResolution(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
        transforms.ToTensor(),
        AddNoise(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        PerImageStandardize(),
    ]


def _with_preprocess(steps: list, preprocess: str) -> list:
    """Prepend the optional (opt-in) radiograph preprocessing step."""
    fn = get_preprocess(preprocess)
    return steps if fn is None else [fn, *steps]


def build_train_transform(size: int = IMAGE_SIZE, preprocess: str = "none", robust: bool = False):
    steps = _robust_augmentations(size) if robust else _train_augmentations(size)
    return transforms.Compose(_with_preprocess(steps, preprocess))


def build_minority_transform(size: int = IMAGE_SIZE, preprocess: str = "none", robust: bool = False):
    steps = _robust_augmentations(size) if robust else _minority_augmentations(size)
    return transforms.Compose(_with_preprocess(steps, preprocess))


def build_val_transform(size: int = IMAGE_SIZE, preprocess: str = "none", robust: bool = False):
    if robust:  # same per-image normalization as robust training; no augmentation
        return transforms.Compose(_with_preprocess(
            [transforms.Resize((size, size)), transforms.ToTensor(), PerImageStandardize()], preprocess))
    return transforms.Compose(_with_preprocess([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        _normalize(),
    ], preprocess))


def build_tta_transforms(size: int = IMAGE_SIZE):
    base = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        _normalize(),
    ])
    flip = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        _normalize(),
    ])
    return [base, flip]


train_transform = build_train_transform()
minority_transform = build_minority_transform()
val_transform = build_val_transform()

base_tta, tta_flip = build_tta_transforms()
tta_transforms_list = [base_tta, tta_flip]
