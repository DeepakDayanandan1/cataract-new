
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from config import Config
from preprocessing.image_loader import load_image
from preprocessing.pipeline import preprocess_pipeline

class CataractDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, return_steps=False):
        """
        Args:
            root_dir (str): Directory with all the images (e.g., config.RAW_DATA_DIR)
            split (str): 'train' or 'valid'
            transform (callable, optional): Optional transform to be applied on a sample.
            return_steps (bool): If True, returns intermediate preprocessing steps (for visualization)
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.return_steps = return_steps
        self.classes = ['normal', 'cataract']
        self.image_paths = []
        self.labels = []

        self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Split directory not found: {self.root_dir}")

        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_name} not found in {self.root_dir}")
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
            image = load_image(img_path)

            # 2. Preprocessing Pipeline
            # Note: pipeline expects RGB numpy array
            processed_img, steps = preprocess_pipeline(image, target_size=Config.IMAGE_SIZE)
            
            # processed_img is (H, W, 3) float32 [0, 1]
            # Convert to torch tensor (C, H, W)
            tensor_img = torch.tensor(processed_img).permute(2, 0, 1).float()

            if self.transform:
                tensor_img = self.transform(tensor_img)

            if self.return_steps:
                return tensor_img, label, steps, img_path
            
            return tensor_img, label

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            # Return a dummy tensor or handle appropriately (potentially dangerous during training)
            # For robust training, usually we might skip or retry, but strict failure is better for debugging
            raise e

class SlitLampDataset(Dataset):
    def __init__(self, root_dir, transform=None, return_steps=False):
        """
        Args:
            root_dir (str): Directory with class folders (e.g., config.SLIT_LAMP_DIR)
            transform (callable, optional): Optional transform to be applied on a sample.
            return_steps (bool): If True, returns intermediate preprocessing steps.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.return_steps = return_steps
        self.classes = Config.SLIT_LAMP_CLASSES
        self.image_paths = []
        self.labels = []

        self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_name} not found in {self.root_dir}")
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
            image = load_image(img_path)
            processed_img, steps = preprocess_pipeline(image, target_size=Config.IMAGE_SIZE, use_green_channel=False)
            
            # Stack 3 channels for DenseNet
            if processed_img.shape[-1] != 3:
                 # If preprocessing returns 1 channel or something else, handle it.
                 # Pipeline usually returns (H, W, 3) now (see pipeline.py:75)
                 pass

            tensor_img = torch.tensor(processed_img).permute(2, 0, 1).float()

            if self.transform:
                tensor_img = self.transform(tensor_img)

            if self.return_steps:
                return tensor_img, label, steps, img_path
            
            return tensor_img, label

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            raise e

class MultiClassCataractDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, expansion_factor=1):
        """
        Args:
            image_paths (list): List of file paths to images.
            labels (list): List of corresponding labels.
            transform (callable, optional): Transform to be applied on a sample.
            expansion_factor (int): Multiplier for dataset size (for augmentation).
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.expansion_factor = expansion_factor
        self.original_len = len(self.image_paths)

    def __len__(self):
        return self.original_len * self.expansion_factor

    def __getitem__(self, idx):
        # Map expanded index back to original index
        original_idx = idx % self.original_len
        
        img_path = self.image_paths[original_idx]
        label = self.labels[original_idx]

        try:
            image = load_image(img_path)
            # Use binary=False or implicit handling if pipeline is generic.
            # Assuming pipeline returns an image suitable for the model input size.
            # We must ensure pipeline doesn't do things specific to binary if not needed, 
            # but currently reuse preprocessing_pipeline.
            processed_img, _ = preprocess_pipeline(image, target_size=Config.IMAGE_SIZE)
            
            tensor_img = torch.tensor(processed_img).permute(2, 0, 1).float()

            if self.transform:
                tensor_img = self.transform(tensor_img)

            return tensor_img, label

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            raise e
