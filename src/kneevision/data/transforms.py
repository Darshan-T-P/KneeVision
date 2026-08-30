from torchvision import transforms

from kneevision.config.settings import IMAGE_SIZE

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


def build_train_transform(size: int = IMAGE_SIZE):
    return transforms.Compose(_train_augmentations(size))


def build_minority_transform(size: int = IMAGE_SIZE):
    return transforms.Compose(_minority_augmentations(size))


def build_val_transform(size: int = IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        _normalize(),
    ])


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
