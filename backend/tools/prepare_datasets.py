
import os
import argparse
import shutil
import cv2
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image

# Setup absolute imports by adding project root to sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.config import Config
from backend.ml.preprocessing.pipeline import preprocess_pipeline
from backend.ml.preprocessing.augmentations import get_offline_aug_transforms


def ensure_clean_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def save_image(image_array, path):
    """Saves a numpy image array (RGB, [0, 1] float or [0, 255] uint8) to path."""
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    Image.fromarray(image_array).save(path)


def collect_raw_images(raw_root):
    """
    Walks raw_root looking for class sub-folders and returns (paths, labels).
    Handles both flat structure (root/class/img.jpg) and pre-split structure
    (root/train/class/img.jpg + root/valid/class/img.jpg) by merging everything.
    """
    # Detect pre-split structure
    has_train = os.path.exists(os.path.join(raw_root, 'train'))
    has_valid = os.path.exists(os.path.join(raw_root, 'valid'))

    if has_train and has_valid:
        print("  Detected pre-split train/valid structure -- merging all images before re-splitting.")
        search_roots = [
            os.path.join(raw_root, 'train'),
            os.path.join(raw_root, 'valid'),
        ]
        test_dir = os.path.join(raw_root, 'test')
        if os.path.exists(test_dir):
            search_roots.append(test_dir)
    else:
        search_roots = [raw_root]

    all_data = []  # [(path, class_name)]
    img_exts = ('.png', '.jpg', '.jpeg')

    for search_root in search_roots:
        for cls in os.listdir(search_root):
            cls_dir = os.path.join(search_root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(img_exts):
                    all_data.append((os.path.join(cls_dir, fname), cls))

    if not all_data:
        # Fallback: recursive walk, label = first subdir name under raw_root
        for root_dir, dirs, files in os.walk(raw_root):
            for f in files:
                if f.lower().endswith(img_exts):
                    path = os.path.join(root_dir, f)
                    rel = os.path.relpath(path, raw_root)
                    parts = rel.split(os.sep)
                    label = parts[0]
                    all_data.append((path, label))

    paths = [x[0] for x in all_data]
    labels = [x[1] for x in all_data]
    return paths, labels


def process_and_save_dataset(
    name,
    raw_root,
    processed_root,
    train_ratio=0.7,
    use_green_channel=True,
    dry_run=False,
):
    """
    New pipeline (correct order):
      1. Collect all raw image paths.
      2. Preprocess EVERY image (resize -> denoise -> CLAHE -> normalize).
      3. Split preprocessed images into train / val  (stratified, seeded).
      4. Save val images as-is.
      5. Save train images + generate Config.AUGMENTATION_FACTOR augmented
         variants per image (total = 1 original + FACTOR augmented copies).
    """
    print(f"\n{'='*55}")
    print(f"  Processing: {name}")
    print(f"{'='*55}")

    if not os.path.exists(raw_root):
        print(f"  Skipping {name}: raw directory not found at {raw_root}")
        return

    # ── Step 1: Collect ──────────────────────────────────────────────────────
    paths, labels = collect_raw_images(raw_root)
    if not paths:
        print("  No images found -- skipping.")
        return

    unique_classes = sorted(set(labels))
    print(f"  Found {len(paths)} raw images across classes: {unique_classes}")

    # ── Step 2: Preprocess everything in memory ───────────────────────────────
    print("  Step 2/4 -- Preprocessing all images ...")
    preprocessed = []   # list of (np_array_uint8, label)
    failed = 0

    iterator = enumerate(zip(paths, labels))
    if not dry_run:
        iterator = enumerate(tqdm(zip(paths, labels), total=len(paths)))

    for idx, (img_path, label) in iterator:
        if dry_run and idx >= 10:
            break
        try:
            pil_img = Image.open(img_path).convert('RGB')
            img_np = np.array(pil_img)

            processed_float, _ = preprocess_pipeline(
                img_np,
                target_size=Config.TARGET_SIZE,
                use_green_channel=use_green_channel,
            )

            img_uint8 = (processed_float * 255).astype(np.uint8)
            preprocessed.append((img_uint8, label))

        except Exception as e:
            print(f"  [WARN] Failed to process {img_path}: {e}")
            failed += 1

    print(f"  Preprocessed {len(preprocessed)} images successfully ({failed} failed).")

    # ── Step 3: Stratified train / val split ─────────────────────────────────
    val_ratio = round(1.0 - train_ratio, 4)
    print(f"  Step 3/4 -- Splitting: train={train_ratio:.0%} / val={val_ratio:.0%} (stratified, seed={Config.SEED})")

    pp_images = [x[0] for x in preprocessed]
    pp_labels = [x[1] for x in preprocessed]

    indices = list(range(len(pp_images)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        stratify=pp_labels,
        random_state=Config.SEED,
    )

    train_items = [(pp_images[i], pp_labels[i]) for i in train_idx]
    val_items   = [(pp_images[i], pp_labels[i]) for i in val_idx]

    print(f"  Split -> Train: {len(train_items)}, Val: {len(val_items)}")

    # ── Step 4: Save val images (no augmentation) ────────────────────────────
    print("  Step 4/4 -- Saving val images ...")
    val_dir = os.path.join(processed_root, 'val')
    counts = {'train': 0, 'val': 0}

    for idx, (img_array, label) in enumerate(tqdm(val_items, desc="  Val")):
        save_dir = os.path.join(val_dir, label)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"val_{idx:05d}.jpg")
        save_image(img_array, save_path)
        counts['val'] += 1

    # ── Step 5: Save train images + augmented copies ──────────────────────────
    print("  Step 5/4 -- Saving train images + augmented copies ...")
    train_dir_out = os.path.join(processed_root, 'train')

    img_type  = 'slit_lamp' if 'slit' in name.lower() else 'fundus'
    aug_level = 'standard'

    transform = None
    if Config.AUGMENTATION_ENABLED:
        transform = get_offline_aug_transforms(image_type=img_type, augmentation_level=aug_level)

    for idx, (img_array, label) in enumerate(tqdm(train_items, desc="  Train")):
        save_dir = os.path.join(train_dir_out, label)
        os.makedirs(save_dir, exist_ok=True)

        base_name = f"train_{idx:05d}"

        # 1. Save original preprocessed
        save_path = os.path.join(save_dir, f"{base_name}_orig.jpg")
        save_image(img_array, save_path)
        counts['train'] += 1

        # 2. Augmented copies (applied to PIL image before saving)
        if Config.AUGMENTATION_ENABLED and transform is not None:
            pil_img = Image.fromarray(img_array)   # uint8 RGB PIL image
            for aug_i in range(Config.AUGMENTATION_FACTOR):
                aug_pil = transform(pil_img)        # PIL -> PIL (all transforms are PIL-based)
                aug_array = np.array(aug_pil)
                aug_path = os.path.join(save_dir, f"{base_name}_aug_{aug_i}.jpg")
                save_image(aug_array, aug_path)
                counts['train'] += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  OK {name} done.")
    print(f"    Val images   : {counts['val']}")
    print(f"    Train images : {counts['train']}", end="")
    if Config.AUGMENTATION_ENABLED:
        orig_count = len(train_items)
        factor = counts['train'] / orig_count if orig_count > 0 else 0
        print(f"  ({orig_count} originals × {factor:.1f}x expansion)")
    else:
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets: preprocess -> split -> augment train"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['binary', 'multiclass', 'slit_lamp', 'all'],
        default='slit_lamp',
        help="Which dataset to process",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of images allocated to training (remainder -> val)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable offline augmentation (overrides Config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only a tiny subset for quick testing",
    )

    args = parser.parse_args()

    if args.no_augment:
        Config.AUGMENTATION_ENABLED = False
        print("Augmentation DISABLED via --no-augment flag.")

    print("=" * 55)
    print("  DATASET PREPARATION  (preprocess -> split -> augment)")
    print("=" * 55)
    print(f"  Target        : {args.dataset}")
    print(f"  Train ratio   : {args.train_ratio:.0%}")
    print(f"  Val ratio     : {(1 - args.train_ratio):.0%}")
    print(f"  Augmentation  : {'ENABLED' if Config.AUGMENTATION_ENABLED else 'DISABLED'}"
          + (f" ({Config.AUGMENTATION_FACTOR}x per image)" if Config.AUGMENTATION_ENABLED else ""))
    print(f"  Seed          : {Config.SEED}")
    print("=" * 55)

    # ── Slit Lamp ─────────────────────────────────────────────────────────────
    if args.dataset in ['slit_lamp', 'all']:
        sl_out = os.path.join(Config.PROCESSED_DATA_DIR, 'slitlamp')
        # Clean output directory first
        if os.path.exists(sl_out):
            print(f"\n  Clearing existing output: {sl_out}")
            shutil.rmtree(sl_out)
        os.makedirs(sl_out, exist_ok=True)

        process_and_save_dataset(
            name="Slit Lamp",
            raw_root=Config.RAW_DATA_SLIT_LAMP,
            processed_root=sl_out,
            train_ratio=args.train_ratio,
            use_green_channel=False,
            dry_run=args.dry_run,
        )

    # ── Fundus Binary ─────────────────────────────────────────────────────────
    if args.dataset in ['binary', 'all']:
        bin_out = os.path.join(Config.PROCESSED_DATA_DIR, 'fundus', 'binary')
        if os.path.exists(bin_out):
            print(f"\n  Clearing existing output: {bin_out}")
            shutil.rmtree(bin_out)
        os.makedirs(bin_out, exist_ok=True)

        process_and_save_dataset(
            name="Fundus Binary",
            raw_root=Config.RAW_DATA_BINARY,
            processed_root=bin_out,
            train_ratio=args.train_ratio,
            use_green_channel=True,
            dry_run=args.dry_run,
        )

    # ── Fundus Multiclass ─────────────────────────────────────────────────────
    if args.dataset in ['multiclass', 'all']:
        mc_out = os.path.join(Config.PROCESSED_DATA_DIR, 'fundus', 'multiclass')
        if os.path.exists(mc_out):
            print(f"\n  Clearing existing output: {mc_out}")
            shutil.rmtree(mc_out)
        os.makedirs(mc_out, exist_ok=True)

        process_and_save_dataset(
            name="Fundus Multiclass",
            raw_root=Config.RAW_DATA_MULTICLASS,
            processed_root=mc_out,
            train_ratio=args.train_ratio,
            use_green_channel=True,
            dry_run=args.dry_run,
        )

    print("\n" + "=" * 55)
    print("  PREPARATION COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    main()
