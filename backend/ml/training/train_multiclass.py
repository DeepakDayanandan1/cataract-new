
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

def train_one_epoch(model, loader, criterion, optimizer, device, dry_run=False):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    loop = tqdm(loader, leave=False)
    for idx, (images, labels) in enumerate(loop):
        if dry_run and idx >= 2:
            break
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

def evaluate(model, loader, criterion, device, dry_run=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            if dry_run and idx >= 2:
                break
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
    # Point explicitly to the new processed dataset folder
    dataset_dir = os.path.join(Config.PROCESSED_DATA_DIR, 'fundus', 'multiclass')
    
    # We no longer need to split manually or use expansion_factor here, 
    # as the data preparation script handled splitting and augmentation.
    
    # Train
    train_ds = MultiClassCataractDataset(
        root_dir=dataset_dir,
        split='train',
        transform=get_train_transforms(image_type='fundus', augmentation_level='very_aggressive'), # Modified for aggressive
        is_preprocessed=True
    )
    
    # Valid
    valid_ds = MultiClassCataractDataset(
        root_dir=dataset_dir,
        split='val',
        transform=get_valid_transforms(image_type='fundus'), # Valid stays simple
        is_preprocessed=True
    )
    
    # Test
    test_ds = MultiClassCataractDataset(
        root_dir=dataset_dir,
        split='test',
        transform=get_valid_transforms(image_type='fundus'),
        is_preprocessed=True
    )
    
    print(f"Train Dataset Length (with expansion): {len(train_ds)}")
    
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION (MULTICLASS)")
    print("="*50)
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {Config.MULTICLASS_BATCH_SIZE}")
    print(f"Learning Rate: {Config.MULTICLASS_LEARNING_RATE}")
    print(f"Augmentation: {'Online + Offline' if Config.AUGMENTATION_ENABLED else 'Online Only'}")
    print(f"Dataset: Fundus Multiclass")
    print("-" * 30)
    print(f"Train Size: {len(train_ds)}")
    print(f"Valid Size: {len(valid_ds)}")
    print(f"Test Size:  {len(test_ds)}")
    print("="*50 + "\n")
    
    train_loader = DataLoader(train_ds, batch_size=Config.MULTICLASS_BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size=Config.MULTICLASS_BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=Config.MULTICLASS_BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    if args.dry_run:
        print("Dry run mode: limiting epochs and batches usually, but here just running 1 epoch.")
        args.epochs = 1

    # 2. Model
    # DenseNet169 with 4 classes (Note: config name might be densenet121 or 169 based on user changes, ensuring consistnecy)
    model = get_model(num_classes=4, dropout_rate=Config.MULTICLASS_DROPOUT_RATE)
    model = model.to(device)
    
    # Layer Freezing - Freeze first 277 parameters (roughly 35% trainable later) (Actually param list order)
    # A better way is freezing by named modules, but let's stick to the user's specific "277 parameters" request or interpreation.
    # Usually "freeze layers" means blocks. DenseNet169 has many laeyrs.
    # If request is freeze_layers: 277. Let's assume this means parameters/modules.
    # Let's count params.
    # To be safe and precise with the request "freeze_layers: 277", I will freeze the first 277 parameters found in model.parameters().
    params = list(model.parameters())
    print(f"Total parameter tensors: {len(params)}")
    freeze_count = 277
    if freeze_count > len(params):
        print(f"Warning: Requested freeze count {freeze_count} > total params {len(params)}. Freezing all except last layer.")
        freeze_count = len(params) - 2 # Keep classifier
        
    for i, param in enumerate(params):
        if i < freeze_count:
            param.requires_grad = False
            
    print(f"Freezing complete. First {freeze_count} parameter groups frozen.")
    
    # Class Weights
    class_weights = torch.tensor(Config.CLASS_WEIGHTS).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer - AdamW
    # Only optimize parameters that require grad
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=Config.MULTICLASS_LEARNING_RATE, 
                            weight_decay=Config.WEIGHT_DECAY)
                            
    # Scheduler - CosineAnnealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.MULTICLASS_EPOCHS)
    
    best_f1 = 0.0 # Use F1 or Loss or Acc to pick best model? Let's use F1 macro.
    
    # 3. Training
    for epoch in range(args.epochs):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch+1}/{args.epochs} | LR: {current_lr:.2e}")
        
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.dry_run
        )
        
        valid_loss, valid_acc, valid_prec, valid_rec, valid_f1 = evaluate(
            model, valid_loader, criterion, device, args.dry_run
        )
        
        # Step Scheduler
        scheduler.step()
        
        print(f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} F1={train_f1:.4f}")
        print(f"Valid: Loss={valid_loss:.4f} Acc={valid_acc:.4f} F1={valid_f1:.4f}")
        
        # Save Best Model based on Valid F1
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            
            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            unique_save_path = os.path.join(Config.MODEL_SAVE_DIR, f"densenet_multiclass_best_{timestamp}.pth")
            standard_save_path = os.path.join(Config.MODEL_SAVE_DIR, "densenet_multiclass_best.pth")
            
            torch.save(model.state_dict(), unique_save_path)
            torch.save(model.state_dict(), standard_save_path)
            
            print(f"Model saved to {unique_save_path} and updated {standard_save_path}")
            
    print("\nTraining Complete. Evaluating on Test Set with Best Model...")
    
    # Load best model
    save_path = os.path.join(Config.MODEL_SAVE_DIR, "densenet_multiclass_best.pth")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print("Loaded best model.")
    else:
        print("Best model not found (maybe first epoch failed?), using current weights.")
        
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
        model, test_loader, criterion, device, args.dry_run
    )
    
    print(f"Test Results: Loss={test_loss:.4f}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=Config.MULTICLASS_EPOCHS, help="Number of epochs")
    parser.add_argument("--dry-run", action="store_true", help="Run a single short epoch")
    args = parser.parse_args()
    main(args)
