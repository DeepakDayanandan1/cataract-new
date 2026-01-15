
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
    Enhances local contrast.
    If image is RGB, converts to LAB, applies to L channel, converts back.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if len(image.shape) == 3: # RGB Image
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else: # Grayscale
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

def preprocess_pipeline(image, target_size=(224, 224), use_green_channel=True):
    """
    Full preprocessing pipeline.
    
    Args:
        image: Input RGB image
        target_size: (height, width)
        use_green_channel: If True, extracts green channel (Fundus). 
                           If False, uses full RGB (Slit-lamp).
    
    Returns:
        processed_image: The final image ready for the model (H, W, 3)
        intermediate_steps: Dictionary containing images at each step for visualization
    """
    steps = {}
    
    # 1. Resize
    resized = resize_image(image, size=target_size)
    steps['resized'] = resized
    
    # 2. Green Channel Extraction (Conditional)
    if use_green_channel:
        current_img = extract_green_channel(resized)
        steps['green_channel'] = current_img
    else:
        current_img = resized
        steps['green_channel'] = resized # Store RGB here for UI compatibility
    
    # 3. Noise Reduction
    denoised = remove_noise(current_img)
    steps['denoised'] = denoised
    
    # 4. CLAHE
    enhanced = apply_clahe(denoised)
    steps['enhanced'] = enhanced
    
    # 5. Normalization
    normalized = normalize_image(enhanced)
    
    # Final Channel Stacking
    if use_green_channel:
        # (H, W) -> (H, W, 3)
        final_input = np.stack((normalized,)*3, axis=-1)
    else:
        # Already (H, W, 3)
        final_input = normalized
    
    return final_input, steps
