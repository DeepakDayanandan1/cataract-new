
import torch
from torchvision import transforms


# ImageNet statistics -- required for pretrained DenseNet169
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Private helpers: return the PIL-based augmentation op lists
# ---------------------------------------------------------------------------

def _slit_lamp_aug_ops():
    return [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),       # slit lamp images are upright
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1), translate=(0.05, 0.05)),
    ]


def _fundus_aug_ops(augmentation_level='standard'):
    if augmentation_level == 'very_aggressive':
        return [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ]
    else:
        return [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_offline_aug_transforms(image_type='fundus', augmentation_level='standard'):
    """
    PIL-only augmentation for OFFLINE disk saving in prepare_datasets.py.

    Input  : PIL.Image (uint8 RGB)
    Output : PIL.Image (uint8 RGB, augmented)

    Does NOT include ToTensor or Normalize. The result is saved to JPEG.
    Do NOT use this at training time -- use get_train_transforms() instead.
    """
    if image_type == 'fundus':
        ops = _fundus_aug_ops(augmentation_level)
    elif image_type == 'slit_lamp':
        ops = _slit_lamp_aug_ops()
    else:
        raise ValueError(f"Unknown image_type: {image_type!r}")

    return transforms.Compose(ops)   # PIL in -> PIL out


def get_train_transforms(image_type='fundus', augmentation_level='standard'):
    """
    Online training transform pipeline (used by dataset.py at training time).

    Input  : PIL.Image (uint8 RGB, already preprocessed/saved to 224x224)
    Output : torch.Tensor [3, 224, 224], ImageNet-normalised float

    PIL augmentations applied first, then ToTensor, then ImageNet Normalize.
    """
    if image_type == 'fundus':
        ops = _fundus_aug_ops(augmentation_level)
    elif image_type == 'slit_lamp':
        ops = _slit_lamp_aug_ops()
    else:
        raise ValueError(f"Unknown image_type: {image_type!r}")

    return transforms.Compose(ops + [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_valid_transforms(image_type='fundus'):
    """
    Validation / test transform pipeline (no augmentation).

    Input  : PIL.Image (uint8 RGB, already preprocessed to 224x224)
    Output : torch.Tensor [3, 224, 224], ImageNet-normalised float
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
