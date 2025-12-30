
import cv2
import numpy as np

def extract_green_channel(image):
    """
    Extracts the Green channel from an RGB image.
    Medical Relevance: The green channel usually provides the best contrast 
    for visualizing retinal structures and opacities (cataracts) because 
    the retinal pigment epithelium absorbs red light and the lens absorbs blue light.
    """
    if len(image.shape) != 3:
        raise ValueError("Input image must be RGB.")
        
    return image[:, :, 1]

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances local contrast, making details in the eye more visible.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def normalize_image(image):
    """
    Normalizes pixel values to [0, 1].
    """
    return image.astype(np.float32) / 255.0

def resize_image(image, size=(224, 224)):
    """
    Resizes the image to the target size.
    """
    return cv2.resize(image, size)

def remove_noise(image):
    """
    Applies Gaussian Blur to reduce high-frequency noise.
    """
    return cv2.GaussianBlur(image, (5, 5), 0)

def preprocess_pipeline(image, target_size=(224, 224)):
    """
    Full preprocessing pipeline:
    Resize -> Green Channel -> Noise Reduction -> CLAHE -> Normalize
    
    Returns:
        processed_image: The final image ready for the model (H, W) or (H, W, C)
        intermediate_steps: Dictionary containing images at each step for visualization
    """
    steps = {}
    
    # 1. Resize
    resized = resize_image(image, size=target_size)
    steps['resized'] = resized
    
    # 2. Green Channel Extraction
    green = extract_green_channel(resized)
    steps['green_channel'] = green
    
    # 3. Noise Reduction (Optional but recommended)
    denoised = remove_noise(green)
    steps['denoised'] = denoised
    
    # 4. CLAHE
    enhanced = apply_clahe(denoised)
    steps['enhanced'] = enhanced
    
    # 5. Normalization
    normalized = normalize_image(enhanced)
    # Stack to 3 channels for DenseNet input (it implies 3 channels usually)
    # We can replicate the channel 3 times or train on 1 channel. 
    # Transfer learning models expect 3 channels.
    final_input = np.stack((normalized,)*3, axis=-1)
    
    return final_input, steps
