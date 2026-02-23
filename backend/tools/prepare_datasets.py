
import os
import argparse
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
from backend.ml.preprocessing.augmentations import get_train_transforms
import torch
from torchvision import transforms

def ensure_clean_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def save_image(image_array, path):
    """Saves a numpy image array (RGB, [0, 1] or [0, 255]) to path."""
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    Image.fromarray(image_array).save(path)

# Removed internal generate_augmentations


def process_and_save_dataset(name, raw_root, processed_root, split_ratios=(0.7, 0.15, 0.15), use_green_channel=True, dry_run=False):
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
            if dry_run and idx >= 2: # Process only 2 images per split for dry run
                break

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
                    # Determine image type and level
                    img_type = 'slit_lamp' if 'slit' in name.lower() else 'fundus'
                    aug_level = 'standard' # Using standard for offline generation to avoid excessive distortion
                    
                    transform = get_train_transforms(image_type=img_type, augmentation_level=aug_level)
                    
                    # 1. Save Original
                    save_name = f"{base_name}_orig.jpg"
                    save_path = os.path.join(save_dir, save_name)
                    save_image(img_array_uint8, save_path)
                    counts[split_name] += 1
                    
                    # 2. Generate Augmented Versions
                    # Generate Config.AUGMENTATION_FACTOR (default 4) additional versions
                    # Or should total be 4? Previous logic generated 4 total (1 orig + 3 augs technically, but logic listed 4).
                    # Config.AUGMENTATION_FACTOR is 4. Let's aim for 4 augmented + 1 original = 5x? 
                    # Previous logic: returns list of 4 images (Original, Rot, Flip, Jitter).
                    # So let's generate 3 augmented versions to match the 4x total if we want exact parity, 
                    # or just use loop for FACTOR amount.
                    # Let's generate Config.AUGMENTATION_FACTOR variations.
                    
                    pil_img = Image.fromarray(img_array_uint8)
                    
                    for i in range(Config.AUGMENTATION_FACTOR):
                        # Apply transform pipeline
                        aug_img = transform(pil_img)
                        # Transform returns PIL Image (usually) or Tensor.
                        # Our augmentations.py returns Compose of PIL transforms, so it returns PIL Image.
                        
                        if isinstance(aug_img, torch.Tensor):
                            aug_img = aug_img.permute(1, 2, 0).numpy() * 255
                            aug_img = aug_img.astype(np.uint8)
                        else:
                            aug_img = np.array(aug_img)
                            
                        save_name = f"{base_name}_aug_{i}.jpg"
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
    print(f"Finished {name}. Counts: {counts}")
    if Config.AUGMENTATION_ENABLED:
        original_train = len(train_paths)
        augmented_train = counts['train']
        expansion_ratio = augmented_train / original_train if original_train > 0 else 0
        print(f"Train Expansion: {original_train} -> {augmented_train} images")
        print(f"Expansion Factor: {expansion_ratio:.2f}x")
    else:
        print(f"Train Count: {len(train_paths)} (No augmentation)")

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for training (Split, Preprocess, Augment)")
    parser.add_argument("--dataset", type=str, choices=['binary', 'multiclass', 'slit_lamp', 'all'], 
                        default='all', help="Which dataset to process")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation (overrides Config)")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick test processing only 1 batch/image per class")
    
    args = parser.parse_args()
    
    if args.no_augment:
        Config.AUGMENTATION_ENABLED = False
        print("Augmentation DISABLED via CLI flag.")
        
    print(f"=== Offline Dataset Preparation ===")
    print(f"Target Dataset(s): {args.dataset}")
    print(f"Augmentation Enabled: {Config.AUGMENTATION_ENABLED}")
    
    # 1. Fundus Binary
    if args.dataset in ['binary', 'all']:
        process_and_save_dataset(
            name="Fundus Binary",
            raw_root=Config.RAW_DATA_BINARY,
            processed_root=os.path.join(Config.PROCESSED_DATA_DIR, 'fundus', 'binary'),
            use_green_channel=True,
            dry_run=args.dry_run
        )
    
    # 2. Fundus Multiclass
    if args.dataset in ['multiclass', 'all']:
        process_and_save_dataset(
            name="Fundus Multiclass",
            raw_root=Config.RAW_DATA_MULTICLASS,
            processed_root=os.path.join(Config.PROCESSED_DATA_DIR, 'fundus', 'multiclass'),
            use_green_channel=True,
            dry_run=args.dry_run
        )
    
    # 3. Slit Lamp
    if args.dataset in ['slit_lamp', 'all']:
        process_and_save_dataset(
            name="Slit Lamp",
            raw_root=Config.RAW_DATA_SLIT_LAMP,
            processed_root=os.path.join(Config.PROCESSED_DATA_DIR, 'slitlamp'),
            use_green_channel=False,
            dry_run=args.dry_run
        )
    
    print("\n" + "="*50)
    print("DATASET PREPARATION COMPLETE")
    print("="*50)
    print(f"Dataset(s) Processed: {args.dataset}")
    print(f"Augmentation Status: {'ENABLED' if Config.AUGMENTATION_ENABLED else 'DISABLED'}")
    if Config.AUGMENTATION_ENABLED:
        print(f"Augmentation Factor: {Config.AUGMENTATION_FACTOR}x (Total 1 Original + {Config.AUGMENTATION_FACTOR} Augmented)")
    
    print("\npreprocessing Steps Applied:")
    print("  [Common]: Resize to (224, 224), Normalize [0,1]")
    print("  [Fundus]: Green Channel Extraction -> Denoise -> CLAHE")
    print("  [Slit Lamp]: Denoise -> LAB Color Conversion -> CLAHE (L-Channel) -> RGB Conversion")
    print("="*50)

if __name__ == "__main__":
    main()
