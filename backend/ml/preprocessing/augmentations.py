
import torch
from torchvision import transforms

def get_train_transforms(image_type='fundus', augmentation_level='standard'):
    """
    Returns training transformations based on image type.
    
    Args:
        image_type (str): 'fundus' or 'slit_lamp'
        augmentation_level (str): 'standard' or 'very_aggressive'
    """
    if image_type == 'fundus':
        if augmentation_level == 'very_aggressive':
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)), # Zoom in/out
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ])
        else:
            # Standard
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
    elif image_type == 'slit_lamp':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # Slit lamp images are usually upright, so limited rotation
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1), translate=(0.05, 0.05)),
        ])
    else:
        raise ValueError(f"Unknown image_type: {image_type}")

def get_valid_transforms(image_type='fundus'):
    """
    Returns validation transformations (usually just normalization if needed, or identity).
    """
    # Since dataset already converts to tensor and pipeline normalizes to [0,1],
    # validation transforms might just be empty or specific normalization.
    return transforms.Compose([
        # No test-time augmentation for now
    ])
