
import os
import shutil
import cv2
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance

# Setup absolute imports by adding project root to sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.config import Config
from backend.ml.preprocessing.pipeline import preprocess_pipeline

def ensure_clean_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def save_image(image_array, path):
    """Saves a numpy image array (RGB, [0, 1] or [0, 255]) to path."""
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    Image.fromarray(image_array).save(path)

def generate_augmentations(image):
    """
    Generates 4 versions of the image:
    1. Original
    2. Random Rotation (-15 to 15 degrees)
    3. Random Flip (Horizontal)
    4. Color Jitter (Brightness/Contrast)
    
    Args:
        image: Numpy array (H, W, 3), uint8 or float [0,1]
    Returns:
        List of 4 numpy images
    """
    # Convert to PIL for easy transformations
    if image.dtype != np.uint8:
        pil_img = Image.fromarray((image * 255).astype(np.uint8))
    else:
        pil_img = Image.fromarray(image)
        
    augs = []
    
    # 1. Original
    augs.append(np.array(pil_img))
    
    # 2. Rotation
    angle = random.uniform(-15, 15)
    rot_img = pil_img.rotate(angle, resample=Image.BILINEAR)
    augs.append(np.array(rot_img))
    
    # 3. Horizontal Flip
    flip_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    augs.append(np.array(flip_img))
    
    # 4. Color Jitter (Brightness/Contrast)
    # Factor 1.0 is original, 0.5 is 50%, 1.5 is 150%
    enhancer = ImageEnhance.Brightness(pil_img)
    bright_img = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Contrast(bright_img)
    contrast_img = enhancer.enhance(random.uniform(0.8, 1.2))
    augs.append(np.array(contrast_img))
    
    # Ensure all are numpy (H, W, 3)
    return augs

def process_and_save_dataset(name, raw_root, processed_root, split_ratios=(0.7, 0.15, 0.15), use_green_channel=True):
    print(f"\nScanning {name} dataset from {raw_root}...")
    
    if not os.path.exists(raw_root):
        print(f"Skipping {name}: Raw directory not found.")
        return

    # Check if 'train' and 'valid' folders exist (Pre-split structure)
    # Specific logic for Binary dataset which might be pre-split
    pre_split = False
    if os.path.exists(os.path.join(raw_root, "train")) and os.path.exists(os.path.join(raw_root, "valid")):
         # Structure: root/train/class, root/valid/class
         print("Detected existing train/valid splits.")
         pre_split = True
         
         # We need to map this to our split logic
         # Train -> Train
         # Valid -> Val
         # Valid -> Test (split valid into val/test? Or just keep valid? User requested Train/Val/Test)
         # If existing structure only has Train/Val, we might just split Val into Val/Test.
         # Or for simplicity, use Valid as Val and Valid as Test (duplicate? No bad).
         # Let's read all from Train and Valid, merge them (conceptually), or just respect them.
         # Let's respect Train. Split Valid into Valid/Test (50/50).
    
    # Logic to Collect Paths
    # If Pre-split:
    if pre_split:
        # Collect Train
        train_paths, train_labels = [], []
        train_dir = os.path.join(raw_root, "train")
        for cls in os.listdir(train_dir):
            cd = os.path.join(train_dir, cls)
            if os.path.isdir(cd):
                for f in os.listdir(cd):
                    train_paths.append(os.path.join(cd, f))
                    train_labels.append(cls)
        
        # Collect Valid/Test from 'valid'
        valid_raw_paths, valid_raw_labels = [], []
        valid_dir = os.path.join(raw_root, "valid")
        for cls in os.listdir(valid_dir):
            cd = os.path.join(valid_dir, cls)
            if os.path.isdir(cd):
                for f in os.listdir(cd):
                    valid_raw_paths.append(os.path.join(cd, f))
                    valid_raw_labels.append(cls)
                    
        # Split 'valid' into val and test
        valid_paths, test_paths, valid_labels, test_labels = train_test_split(
            valid_raw_paths, valid_raw_labels, test_size=0.5, stratify=valid_raw_labels, random_state=Config.SEED
        )
        
    else:
        # Standard Root/Class structure
        classes = [d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))]
        if "Cataract" in classes and "train" in os.listdir(os.path.join(raw_root, "Cataract")):
             # Handle the specific "Cataract/train" structure if it exists
             # This seems to be the case from my tool usage previously
             print("Detected nested Cataract/train structure inside one of the classes, handling recursively?")
             # Actually, if the user said binary is split, and I saw fundus_binary/Cataract/train...
             # It implies structure is fundus_binary/[Class]/[Split]
             # This is inverted standard. 
             # Let's try to parse recursively.
             all_images = [] # (path, label)
             for root, dirs, files in os.walk(raw_root):
                 for f in files:
                     if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                         path = os.path.join(root, f)
                         # Guess label from path?
                         # path: .../raw/fundus_binary/Cataract/train/img.jpg -> Label=Cataract?
                         # path: .../raw/fundus_binary/Normal/[???]/img.jpg -> Label=Normal?
                         # This depends on root folder name relative to raw_root.
                         rel = os.path.relpath(path, raw_root)
                         parts = rel.split(os.sep)
                         # parts[0] is usually Class (Cataract, Normal)
                         label = parts[0] 
                         all_images.append((path, label))
            
             paths = [x[0] for x in all_images]
             labels = [x[1] for x in all_images]
             
             train_paths, test_paths, train_labels, test_labels = train_test_split(paths, labels, test_size=0.3, stratify=labels, random_state=Config.SEED)
             valid_paths, test_paths, valid_labels, test_labels = train_test_split(test_paths, test_labels, test_size=0.5, stratify=test_labels, random_state=Config.SEED)

        else:
            # Normal Root/Class
            all_data = [] 
            for cls in classes:
                cls_dir = os.path.join(raw_root, cls)
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_data.append((os.path.join(cls_dir, fname), cls))
            
            paths = [x[0] for x in all_data]
            labels = [x[1] for x in all_data]
            
            if not paths:
                print("No images found.")
                return

            train_paths, test_paths, train_labels, test_labels = train_test_split(paths, labels, test_size=0.3, stratify=labels, random_state=Config.SEED)
            valid_paths, test_paths, valid_labels, test_labels = train_test_split(test_paths, test_labels, test_size=0.5, stratify=test_labels, random_state=Config.SEED)

    print(f"Split sizes: Train={len(train_paths)}, Valid={len(valid_paths)}, Test={len(test_paths)}")
    
    splits = {
        'train': (train_paths, train_labels),
        'val': (valid_paths, valid_labels),
        'test': (test_paths, test_labels)
    }
    
    counts = {'train': 0, 'val': 0, 'test': 0}
    
    for split_name, (sp_paths, sp_labels) in splits.items():
        print(f"Processing {split_name} set...")
        
        for idx, (img_path, label) in enumerate(tqdm(zip(sp_paths, sp_labels), total=len(sp_paths))):
            try:
                pil_img = Image.open(img_path).convert('RGB')
                img_np = np.array(pil_img)
                
                # Preprocess
                processed_img, _ = preprocess_pipeline(
                    img_np, 
                    target_size=Config.TARGET_SIZE,
                    use_green_channel=use_green_channel
                )
                
                img_array_uint8 = (processed_img * 255).astype(np.uint8) if processed_img.dtype != np.uint8 else processed_img
                
                # Prepare Save
                save_dir = os.path.join(processed_root, split_name, label)
                os.makedirs(save_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Augment Check
                if split_name == 'train' and Config.AUGMENTATION_ENABLED:
                    # Generate Augmentations
                    augs = generate_augmentations(img_array_uint8) # Returns list of 4 (H,W,3) uint8 arrays
                    
                    for i, aug_img in enumerate(augs):
                        save_name = f"{base_name}_{i}.jpg"
                        save_path = os.path.join(save_dir, save_name)
                        save_image(aug_img, save_path)
                        counts[split_name] += 1
                else:
                    # Save Original (Preprocessed)
                    save_name = f"{base_name}.jpg"
                    save_path = os.path.join(save_dir, save_name)
                    save_image(processed_img, save_path)
                    counts[split_name] += 1
                    
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

    print(f"Finished {name}. Counts: {counts}")
    if Config.AUGMENTATION_ENABLED:
        original_train = len(train_paths)
        augmented_train = counts['train']
        print(f"Train Expansion: {original_train} -> {augmented_train} (Factor: {augmented_train/original_train if original_train > 0 else 0:.2f}x)")

def main():
    print("=== Offline Dataset Preparation ===")
    
    # 1. Fundus Binary
    process_and_save_dataset(
        name="Fundus Binary",
        raw_root=Config.RAW_DATA_BINARY,
        processed_root=os.path.join(Config.PROCESSED_DATA_DIR, 'fundus', 'binary'),
        use_green_channel=True
    )
    
    # 2. Fundus Multiclass
    process_and_save_dataset(
        name="Fundus Multiclass",
        raw_root=Config.RAW_DATA_MULTICLASS,
        processed_root=os.path.join(Config.PROCESSED_DATA_DIR, 'fundus', 'multiclass'),
        use_green_channel=True
    )
    
    # 3. Slit Lamp
    process_and_save_dataset(
        name="Slit Lamp",
        raw_root=Config.RAW_DATA_SLIT_LAMP,
        processed_root=os.path.join(Config.PROCESSED_DATA_DIR, 'slitlamp'),
        use_green_channel=False
    )
    
    print("\nDataset Preparation Complete.")

if __name__ == "__main__":
    main()
