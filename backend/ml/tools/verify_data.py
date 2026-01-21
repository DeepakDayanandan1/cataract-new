
import os
import cv2
import sys
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def verify_dataset(root_dir):
    print(f"Verifying dataset at: {root_dir}")
    
    if not os.path.exists(root_dir):
        print("Error: Dataset directory not found!")
        return

    # Check for subdirectories (classes)
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"Classes found: {classes}")
    
    stats = {}
    corrupt_files = []
    
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        files = os.listdir(cls_dir)
        stats[cls] = 0
        
        print(f"Checking class '{cls}' ({len(files)} files)...")
        
        for fname in tqdm(files):
            fpath = os.path.join(cls_dir, fname)
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
                
            try:
                # Try reading with opencv
                img = cv2.imread(fpath)
                if img is None:
                    print(f"Corrupt (None): {fpath}")
                    corrupt_files.append(fpath)
                else:
                    if img.size == 0:
                        print(f"Corrupt (Empty): {fpath}")
                        corrupt_files.append(fpath)
                    else:
                        stats[cls] += 1
                        
            except Exception as e:
                print(f"Corrupt (Exception): {fpath} - {e}")
                corrupt_files.append(fpath)
                
    print("\nDataset Statistics:")
    total = sum(stats.values())
    for cls, count in stats.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  - {cls}: {count} images ({percentage:.2f}%)")
        
    print(f"\nTotal valid images: {total}")
    print(f"Total corrupt images: {len(corrupt_files)}")
    
    if corrupt_files:
        print("\nList of corrupt files:")
        for cf in corrupt_files[:10]:
            print(cf)
        if len(corrupt_files) > 10:
            print(f"...and {len(corrupt_files) - 10} more.")

    # Check balance
    counts = list(stats.values())
    if counts:
        max_count = max(counts)
        min_count = min(counts)
        if min_count > 0 and (max_count / min_count) > 2.0:
            print("\nWARNING: Dataset is heavily imbalanced!")
            print(f"Ratio Max/Min: {max_count/min_count:.2f}")

if __name__ == "__main__":
    # Check both potentially
    print("--- Checking Cataract Dataset (Training Split) ---")
    train_dir = os.path.join(Config.RAW_DATA_DIR, 'train')
    verify_dataset(train_dir)
    
    print("\n--- Checking Dataset Cataract Final (if exists) ---")
    final_dir = os.path.join(os.path.dirname(Config.RAW_DATA_DIR), 'dataset_cataract_final')
    if os.path.exists(final_dir):
        verify_dataset(final_dir)
    else:
        print(f"Path not found: {final_dir}")
