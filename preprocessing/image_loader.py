
import cv2
import os
import numpy as np
from PIL import Image

def load_image(image_path):
    """
    Loads an image from the given path.
    Supports basic error handling and validation.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    try:
        # Load using OpenCV (reads as BGR)
        image = cv2.imread(image_path)
        if image is None:
             raise ValueError(f"Failed to load image at {image_path}. Format might be unsupported.")
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb
    except Exception as e:
        raise RuntimeError(f"Error loading image {image_path}: {str(e)}")

def save_image(image, save_path):
    """
    Saves a numpy array image to disk.
    """
    try:
        # Convert RGB back to BGR for OpenCV saving
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image # Grayscale
            
        cv2.imwrite(save_path, image_bgr)
    except Exception as e:
        raise RuntimeError(f"Error saving image to {save_path}: {str(e)}")
