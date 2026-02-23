import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from backend.config import Config
from backend.ml.preprocessing.dataset import SlitLampDataset
from backend.ml.models.densenet import get_model
from backend.ml.preprocessing.augmentations import get_valid_transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

def check_distribution():
    processed_root = os.path.join(Config.PROCESSED_DATA_DIR, 'slitlamp')
    
    for split in ['train', 'val', 'test']:
        print(f"--- SPLIT: {split} ---")
        dataset = SlitLampDataset(root_dir=processed_root, split=split, transform=get_valid_transforms('slit_lamp'), is_preprocessed=True)
        counts = [0] * Config.SLIT_LAMP_NUM_CLASSES
        for img, lbl in dataset:
            counts[lbl] += 1
        print(f"Indices counts: {counts}")
        print(f"Classes: {dataset.classes}")

def check_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_root = os.path.join(Config.PROCESSED_DATA_DIR, 'slitlamp')
    valid_dataset = SlitLampDataset(root_dir=processed_root, split='val', transform=get_valid_transforms('slit_lamp'), is_preprocessed=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.SLIT_LAMP_BATCH_SIZE, shuffle=False)
    
    model_path = os.path.join(Config.MODEL_SAVE_DIR, f"{Config.SLIT_LAMP_MODEL_NAME}_best.pth")
    print(f"\nLoading model from {model_path}")
    
    # Init clean model
    model = get_model(num_classes=Config.SLIT_LAMP_NUM_CLASSES, dropout_rate=Config.SLIT_LAMP_DROPOUT_RATE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 1. EVAL MODE
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images.to(device))
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print(f"Accuracy in eval() mode: {accuracy_score(all_labels, all_preds)}")
    
    # 2. TRAIN MODE (Checking if BatchNorm is causing issues)
    model.train()
    all_preds_t = []
    all_labels_t = []
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images.to(device))
            preds = torch.argmax(outputs, dim=1)
            all_preds_t.extend(preds.cpu().numpy())
            all_labels_t.extend(labels.numpy())
            
    print(f"Accuracy in train() mode: {accuracy_score(all_labels_t, all_preds_t)}")

if __name__ == "__main__":
    check_distribution()
    check_model()
