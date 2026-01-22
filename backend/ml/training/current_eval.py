
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from sklearn.metrics import classification_report
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from backend.config import Config
from backend.ml.preprocessing.dataset import SlitLampDataset
from backend.ml.models.densenet import get_model
from backend.ml.preprocessing.augmentations import get_valid_transforms

def evaluate_model():
    Config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    full_dataset = SlitLampDataset(root_dir=Config.RAW_DATA_SLIT_LAMP, transform=get_valid_transforms('slit_lamp'))
    
    loader = DataLoader(
        full_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS
    )
    
    print(f"Total samples: {len(full_dataset)}")
    
    # 2. Load Model
    model = get_model(num_classes=Config.SLIT_LAMP_NUM_CLASSES, dropout_rate=Config.DROPOUT_RATE)
    model = model.to(device)
    
    save_path = os.path.join(Config.MODEL_SAVE_DIR, f"{Config.SLIT_LAMP_MODEL_NAME}_best.pth")
    if not os.path.exists(save_path):
        print(f"Model file not found at {save_path}")
        return

    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"Loaded model from {save_path}")
    
    # 3. Evaluate
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        loop = tqdm(loader, desc="Evaluating")
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device).long()
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Generate Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=Config.SLIT_LAMP_CLASSES, digits=4))

if __name__ == "__main__":
    evaluate_model()
