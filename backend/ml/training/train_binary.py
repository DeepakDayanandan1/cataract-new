
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import numpy as np
import sys

# Add project root to path
# backend/ml/training -> backend/ml -> backend -> root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from backend.config import Config
from backend.ml.preprocessing.dataset import CataractDataset
from backend.ml.models.densenet import get_model
from backend.ml.preprocessing.augmentations import get_train_transforms, get_valid_transforms

def train_one_epoch(model, loader, criterion, optimizer, device, dry_run=False):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    loop = tqdm(loader, leave=False)
    for idx, (images, labels) in enumerate(loop):
        if hasattr(args, 'dry_run') and args.dry_run and idx >= 2:
            break
            
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        loop.set_description(f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, loader, criterion, device, dry_run=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            if dry_run and idx >= 2:
                break
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Print Confusion Matrix once per validation
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    
    return epoch_loss, epoch_acc, epoch_f1

def main(args):
    Config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    # Point to processed data
    # IMPORTANT: The CataractDataset now expects root_dir to potentially have splits or classes inside.
    # We structured it as processed/fundus/binary/train and processed/fundus/binary/val
    
    binary_root = os.path.join(Config.PROCESSED_DATA_DIR, 'fundus', 'binary')
    
    # Train set (Augmented ON DISK -> No heavy online transforms needed, maybe just Normalize)
    # Actually, we might still want Normalize if we didn't do it in dataset (we only did ToTensor)
    # But get_train_transforms usually adds Rotation etc. We should use get_valid_transforms (Resize/Recenter/Normalize)
    # or a custom minimal transform.
    # Since dataset.py does ToTensor, we might just need Normalize.
    # For now, let's use get_valid_transforms which is usually lighter (Resize/Normalize), 
    # but our images are already resized.
    # So we used is_preprocessed=True in Dataset, which does ToTensor.
    # Let's inspect Dataset again. It does ToTensor. 
    # Transforms usually expect PIL or Tensor. 
    # If we pass transform to Dataset, it applies it after ToTensor.
    # We should likely just use Normalize here.
    
    train_dataset = CataractDataset(
        root_dir=binary_root, 
        split='train',
        transform=get_train_transforms(image_type='fundus', augmentation_level='standard'),
        is_preprocessed=True
    )
    
    valid_dataset = CataractDataset(
        root_dir=binary_root, 
        split='val', # Note: 'val' not 'valid' per my prepare script
        transform=get_valid_transforms(image_type='fundus'), # Valid stays simple
        is_preprocessed=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BINARY_BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=Config.BINARY_BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS
    )
    
    print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")
    
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {Config.BINARY_BATCH_SIZE}")
    print(f"Learning Rate: {Config.BINARY_LEARNING_RATE}")
    print(f"Augmentation: {'Online + Offline' if Config.AUGMENTATION_ENABLED else 'Online Only'}")
    print(f"Dataset: Fundus Binary")
    print("-" * 30)
    print(f"Train Size: {len(train_dataset)}")
    print(f"Valid Size: {len(valid_dataset)}")
    print("="*50 + "\n")

    if args.dry_run:
        print("Dry run mode enabled. Training for limited batches.")
        args.epochs = 1
    
    # 2. Model, Loss, Optimizer
    model = get_model(num_classes=Config.NUM_CLASSES, dropout_rate=Config.BINARY_DROPOUT_RATE)
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.BINARY_LEARNING_RATE)
    
    best_valid_acc = 0.0
    patience_counter = 0
    
    # 3. Training Loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, args.dry_run)
        valid_loss, valid_acc, valid_f1 = validate(model, valid_loader, criterion, device, args.dry_run)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Valid Loss: {valid_loss:.4f} | Acc: {valid_acc:.4f} | F1: {valid_f1:.4f}")
        
        if valid_f1 > best_valid_acc:
            best_valid_acc = valid_f1 # reuse variable name to avoid refactoring whole logic, effectively best_score
            patience_counter = 0
            
            # Save with timestamp to prevent overwrite
            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            unique_save_path = os.path.join(Config.MODEL_SAVE_DIR, f"{Config.MODEL_NAME}_best_{timestamp}.pth")
            standard_save_path = os.path.join(Config.MODEL_SAVE_DIR, f"{Config.MODEL_NAME}_best.pth")
            
            torch.save(model.state_dict(), unique_save_path)
            torch.save(model.state_dict(), standard_save_path)
            
            print(f"Model saved to {unique_save_path} and updated {standard_save_path} (Best F1)")
        else:
            patience_counter += 1
            
        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break
            
        if args.dry_run:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=Config.BINARY_EPOCHS, help="Number of epochs")
    parser.add_argument("--dry-run", action="store_true", help="Run a single epoch for testing")
    args = parser.parse_args()
    
    # Inject dry_run into args if not present (though it is here) purely for function signature compatibility if needed
    # But better to pass args to train_one_epoch or attach to model? No, simple break loop.
    # We need to pass args or dry_run flag to train_one_epoch to break early.
    # Or just use the global args which is dirty.
    # Let's clean this up: Pass dry_run to train_one_epoch
    main(args)
