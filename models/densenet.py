
import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=1, dropout_rate=0.5, pretrained=True):
    """
    Returns a DenseNet-169 model customized for the task.
    """
    # Load pre-trained DenseNet-169
    # weights='DEFAULT' corresponds to the best available weights (ImageNet)
    weights = models.DenseNet169_Weights.DEFAULT if pretrained else None
    model = models.densenet169(weights=weights)

    # Freeze feature extractor layers (optional, but good for initial transfer learning)
    # Strategy: Freeze all initially, then we can unfreeze later if needed.
    # However, for this task, a common approach is to just train the head 
    # or fine-tune the whole model with a low learning rate.
    # Let's keep it trainable for now as medical images differ from ImageNet.
    
    # Replace the classifier (Linear layer)
    # DenseNet-169 classifier input features is 1664
    num_ftrs = model.classifier.in_features
    
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes)
        # No sigmoid here if using BCEWithLogitsLoss
    )
    
    return model
