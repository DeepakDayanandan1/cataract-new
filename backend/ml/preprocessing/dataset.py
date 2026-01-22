
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from backend.config import Config
from backend.ml.preprocessing.image_loader import load_image
from backend.ml.preprocessing.pipeline import preprocess_pipeline

class CataractDataset(Dataset):
    def __init__(self, root_dir, split=None, transform=None, is_preprocessed=True):
        """
        Args:
            root_dir (str): Directory with the images (e.g. processed/fundus/binary/train)
            split (str): Optional subfolder (e.g. 'train'). If None, assumes root_dir is the split folder.
            transform (callable, optional): Transform (augmentations)
            is_preprocessed (bool): If True, assumes images are already Green/Denoised/CLAHEd. 
                                    Will only do ToTensor (and Normalize if needed).
        """
        if split:
            self.root_dir = os.path.join(root_dir, split)
        else:
            self.root_dir = root_dir
            
        self.transform = transform
        self.is_preprocessed = is_preprocessed
        self.classes = ['normal', 'cataract']
        self.image_paths = []
        self.labels = []

        self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.root_dir):
            if self.root_dir: # Only warn if not empty
                print(f"Warning: Directory not found: {self.root_dir}")
            return

        # Attempt to find class folders directly
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                # Try case insensitive
                found = False
                for existing_dir in os.listdir(self.root_dir):
                    if existing_dir.lower() == class_name.lower():
                        class_dir = os.path.join(self.root_dir, existing_dir)
                        found = True
                        break
                if not found:
                    continue
            
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # 1. Load Image
            if self.is_preprocessed:
                # Load directly as RGB
                image = Image.open(img_path).convert('RGB')
                processed_img = np.array(image)
                # It is already [0, 255] uint8 usually if saved as jpg
                # Convert to [0, 1] float for model
                processed_img = processed_img.astype(np.float32) / 255.0
            else:
                # Use Pipeline
                image = load_image(img_path)
                processed_img, _ = preprocess_pipeline(image, target_size=Config.TARGET_SIZE)

            # Convert to torch tensor (C, H, W)
            tensor_img = torch.tensor(processed_img).permute(2, 0, 1).float()

            if self.transform:
                tensor_img = self.transform(tensor_img)
            
            return tensor_img, label

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            raise e

class SlitLampDataset(Dataset):
    def __init__(self, root_dir, split=None, transform=None, is_preprocessed=True):
        if split:
            self.root_dir = os.path.join(root_dir, split)
        else:
            self.root_dir = root_dir

        self.transform = transform
        self.is_preprocessed = is_preprocessed
        self.classes = Config.SLIT_LAMP_CLASSES
        self.image_paths = []
        self.labels = []

        self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.root_dir):
            return

        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            if self.is_preprocessed:
                image = Image.open(img_path).convert('RGB')
                processed_img = np.array(image)
                processed_img = processed_img.astype(np.float32) / 255.0
            else:
                image = load_image(img_path)
                processed_img, _ = preprocess_pipeline(image, target_size=Config.TARGET_SIZE, use_green_channel=False)

            tensor_img = torch.tensor(processed_img).permute(2, 0, 1).float()

            if self.transform:
                tensor_img = self.transform(tensor_img)
            
            return tensor_img, label

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            raise e

class MultiClassCataractDataset(Dataset):
    def __init__(self, root_dir, split=None, transform=None, is_preprocessed=True):
        # We assume for Multiclass processed data, it follows the same folder structure
        # root_dir/class_name/img.jpg
        # Passing root_dir directly is easiest if we already resolve paths.
        # But to be consistent, let's allow finding classes.
        
        if split:
            self.root_dir = os.path.join(root_dir, split)
        else:
            self.root_dir = root_dir
            
        self.transform = transform
        self.is_preprocessed = is_preprocessed
        # We need to discover classes or use config. But here we usually scan.
        self.classes = ['normal', 'mild', 'moderate', 'severe'] # Or detect
        self.image_paths = []
        self.labels = []

        self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.root_dir):
            return
            
        # If we didn't hardcode classes, we'd listdir. But we know them usually.
        # Let's verify what folders exist
        existing = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        
        # Use existing if they match expected roughly
        # Or just iterate our hardcoded expectation
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            if self.is_preprocessed:
                image = Image.open(img_path).convert('RGB')
                processed_img = np.array(image)
                processed_img = processed_img.astype(np.float32) / 255.0
            else:
                image = load_image(img_path)
                processed_img, _ = preprocess_pipeline(image, target_size=Config.TARGET_SIZE)

            tensor_img = torch.tensor(processed_img).permute(2, 0, 1).float()

            if self.transform:
                tensor_img = self.transform(tensor_img)

            return tensor_img, label

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            raise e
