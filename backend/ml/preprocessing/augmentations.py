
import torch
from torchvision import transforms

def get_train_transforms(image_type='fundus'):
    """
    Returns training transformations based on image type.
    
    Args:
        image_type (str): 'fundus' or 'slit_lamp'
    """
    if image_type == 'fundus':
        return transforms.Compose([
            # Fundus images can be rotated slightly, but usually orientation matters. 
            # Vert/Horiz flips are valid for fundus (retina doesn't change meaning if flipped, usually)
            # However, optic disc location changes. But for classification (cataract vs normal),
            # the features are local texture/opacity.
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # Normalization is often done in dataset or here. 
            # In dataset.py, tensor is created. So these expect Tensor.
            # Normalization to [0,1] happens in pipeline.py.
            # So here we might typically normalize to mean/std if using pretrained models.
            # DenseNet expects mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] usually.
            # Use a safe default for medical images if domain specific mean/std unknown, 
            # or just rely on the [0,1] input. 
            # For now, let's just stick to geometric/color augs.
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
