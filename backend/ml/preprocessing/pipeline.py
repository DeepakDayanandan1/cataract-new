
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

def apply_clahe_color(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE to a color image by converting to LAB space 
    and processing the L (Lightness) channel.
    This avoids color artifacts that occur when processing RGB channels individually.
    """
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    
    # Merge and convert back to RGB
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

def preprocess_pipeline(image, target_size=(224, 224), use_green_channel=True):
    """
    Full preprocessing pipeline.
    
    Args:
        image: Input image (RGB)
        target_size: Desired output size
        use_green_channel: If True, extracts green channel (for fundus). 
                          If False, keeps RGB (for slit lamp).
    
    Returns:
        processed_image: The final image ready for the model (H, W, 3)
        intermediate_steps: Dictionary containing images at each step for visualization
    """
    steps = {}
    
    # 1. Resize
    resized = resize_image(image, size=target_size)
    steps['resized'] = resized
    
    if use_green_channel:
        # 2. Green Channel Extraction
        green = extract_green_channel(resized)
        steps['green_channel'] = green
        
        # 3. Noise Reduction
        denoised = remove_noise(green)
        steps['denoised'] = denoised
        
        # 4. CLAHE
        enhanced = apply_clahe(denoised)
        steps['enhanced'] = enhanced
        
        # 5. Normalization
        normalized = normalize_image(enhanced)
        final_input = np.stack((normalized,)*3, axis=-1)
    else:
        # For Slit Lamp (Color)
        # 2. Noise Reduction (on RGB)
        denoised = remove_noise(resized)
        steps['denoised'] = denoised
        
        # 3. CLAHE (Color-safe via LAB space)
        enhanced = apply_clahe_color(denoised)
        steps['enhanced'] = enhanced
        
        # 4. Normalization
        normalized = normalize_image(enhanced)
        final_input = normalized # Already (H, W, 3)
        
    return final_input, steps
