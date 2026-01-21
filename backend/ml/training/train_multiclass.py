
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from backend.config import Config
from backend.ml.preprocessing.dataset import MultiClassCataractDataset
from backend.ml.models.densenet import get_model
from backend.ml.preprocessing.augmentations import get_train_transforms, get_valid_transforms

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_all_data(root_dir):
    """
    scans root_dir for class folders and returns image paths and labels.
    Assumes classes are: normal, mild, moderate, severe
    """
    # Mapping based on typical severity (ordering matters for consistency if we care, 
    # but for classification just needs to be unique)
    # Let's define specific mapping to ensure consistency
    class_map = {
        'normal': 0,
        'mild': 1,
        'moderate': 2,
        'severe': 3
    }
    
    image_paths = []
    labels = []
    
    # Check which folders exist
    available_classes = []
    for cls in class_map.keys():
        if os.path.exists(os.path.join(root_dir, cls)):
            available_classes.append(cls)
            
    print(f"Found classes: {available_classes}")
    
    for cls in available_classes:
        class_dir = os.path.join(root_dir, cls)
        label = class_map[cls]
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(label)
                
    return image_paths, labels, class_map

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    loop = tqdm(loader, leave=False)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device).long() # Make sure labels are long for CrossEntropy
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        loop.set_description(f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader)
    
    # Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, acc, prec, rec, f1

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).long()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    
    return epoch_loss, acc, prec, rec, f1

def main(args):
    set_seed()
    Config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    # Point explicitly to the new dataset folder
    dataset_dir = Config.RAW_DATA_MULTICLASS
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return

    paths, labels, class_map = load_all_data(dataset_dir)
    print(f"Total images found: {len(paths)}")
    print(f"Class mapping: {class_map}")
    
    # Split 70/15/15
    # First split: 70% Train, 30% Temp (Valid + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=42
    )
    
    # Second split: Split Temp equally (15% Valid, 15% Test of original total)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    print(f"Split sizes: Train={len(X_train)}, Valid={len(X_valid)}, Test={len(X_test)}")
    
    # Create Datasets
    # Train: Expansion factor 4 (4x size effectively via augmentation)
    train_ds = MultiClassCataractDataset(
        X_train, y_train, 
        transform=get_train_transforms('fundus'), 
        expansion_factor=4
    )
    
    valid_ds = MultiClassCataractDataset(
        X_valid, y_valid, 
        transform=get_valid_transforms('fundus'), 
        expansion_factor=1
    )
    
    test_ds = MultiClassCataractDataset(
        X_test, y_test, 
        transform=get_valid_transforms('fundus'), 
        expansion_factor=1
    )
    
    print(f"Train Dataset Length (with expansion): {len(train_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    if args.dry_run:
        print("Dry run mode: limiting epochs and batches usually, but here just running 1 epoch.")
        args.epochs = 1

    # 2. Model
    # DenseNet169 with 4 classes
    model = get_model(num_classes=4, dropout_rate=Config.DROPOUT_RATE)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    best_f1 = 0.0 # Use F1 or Loss or Acc to pick best model? Let's use F1 macro.
    
    # 3. Training
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        valid_loss, valid_acc, valid_prec, valid_rec, valid_f1 = evaluate(
            model, valid_loader, criterion, device
        )
        
        print(f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} F1={train_f1:.4f}")
        print(f"Valid: Loss={valid_loss:.4f} Acc={valid_acc:.4f} F1={valid_f1:.4f}")
        
        # Save Best Model based on Valid F1
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            save_path = os.path.join(Config.MODEL_SAVE_DIR, "densenet_multiclass_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            
    print("\nTraining Complete. Evaluating on Test Set with Best Model...")
    
    # Load best model
    save_path = os.path.join(Config.MODEL_SAVE_DIR, "densenet_multiclass_best.pth")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print("Loaded best model.")
    else:
        print("Best model not found (maybe first epoch failed?), using current weights.")
        
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"Test Results: Loss={test_loss:.4f}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS, help="Number of epochs")
    parser.add_argument("--dry_run", action="store_true", help="Run a single short epoch")
    args = parser.parse_args()
    main(args)
