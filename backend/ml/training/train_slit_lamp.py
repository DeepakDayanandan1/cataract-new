
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from backend.config import Config
from backend.ml.preprocessing.dataset import SlitLampDataset
from backend.ml.models.densenet import get_model
from backend.ml.preprocessing.augmentations import get_train_transforms, get_valid_transforms
from torch.utils.data import Dataset

def train_one_epoch(model, loader, criterion, optimizer, device, dry_run=False):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_confidences = []
    
    loop = tqdm(loader, leave=False)
    for idx, (images, labels) in enumerate(loop):
        if dry_run and idx >= 2:
            break
        images = images.to(device)
        labels = labels.to(device).long() # CrossEntropyLoss expects long labels
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy and confidence
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidences, preds = torch.max(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().detach().numpy())
        
        loop.set_description(f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    
    return epoch_loss, acc, prec, rec, f1, avg_conf

def validate(model, loader, criterion, device, dry_run=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            if dry_run and idx >= 2:
                break
            images = images.to(device)
            labels = labels.to(device).long()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            
    epoch_loss = running_loss / len(loader)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    
    print(f"\nAverage Confidence Score: {avg_conf:.4f}")
    print("Validation Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=Config.SLIT_LAMP_CLASSES, zero_division=0))
    
    return epoch_loss, acc, prec, rec, f1, avg_conf

def main(args):
    Config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    # 1. Prepare Data
    processed_root = os.path.join(Config.PROCESSED_DATA_DIR, 'slitlamp')
    
    # Train
    train_dataset = SlitLampDataset(
        root_dir=processed_root,
        split='train',
        transform=get_train_transforms(image_type='slit_lamp', augmentation_level='standard'),
        is_preprocessed=True
    )
    
    # Valid
    valid_dataset = SlitLampDataset(
        root_dir=processed_root,
        split='val',
        transform=get_valid_transforms(image_type='slit_lamp'),
        is_preprocessed=True
    )
    
    # Note: Slit lamp script didn't have Test set explicitly in original loop, 
    # but our prepare script created one. We can add it or just ignore for now to keep changes minimal.
    # The user asked just to support the new augmentation strategy for training.
    
    # We remove the custom TransformDataset class since we use SlitLampDataset directly.
   
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.SLIT_LAMP_BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=Config.SLIT_LAMP_BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS
    )
    
    print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")
    
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION (SLIT LAMP)")
    print("="*50)
    print(f"Model: {Config.SLIT_LAMP_MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {Config.SLIT_LAMP_BATCH_SIZE}")
    print(f"Learning Rate: {Config.SLIT_LAMP_LEARNING_RATE}")
    print(f"Augmentation: {'Online + Offline' if Config.AUGMENTATION_ENABLED else 'Online Only'}")
    print(f"Dataset: Slit Lamp")
    print("-" * 30)
    print(f"Train Size: {len(train_dataset)}")
    print(f"Valid Size: {len(valid_dataset)}")
    print("="*50 + "\n")

    if args.dry_run:
        print("Dry run mode enabled. Limiting to 1 epoch and few batches.")
        args.epochs = 1
    
    # 2. Model, Loss, Optimizer
    # Config.SLIT_LAMP_NUM_CLASSES is 3
    model = get_model(num_classes=Config.SLIT_LAMP_NUM_CLASSES, dropout_rate=Config.SLIT_LAMP_DROPOUT_RATE)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.SLIT_LAMP_LEARNING_RATE)
    
    best_valid_acc = 0.0
    patience_counter = 0
    
    # 3. Training Loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, train_prec, train_rec, train_f1, train_conf = train_one_epoch(model, train_loader, criterion, optimizer, device, args.dry_run)
        valid_loss, valid_acc, valid_prec, valid_rec, valid_f1, valid_conf = validate(model, valid_loader, criterion, device, args.dry_run)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f} | Conf: {train_conf:.4f}")
        print(f"Valid Loss: {valid_loss:.4f} | Acc: {valid_acc:.4f} | Prec: {valid_prec:.4f} | Rec: {valid_rec:.4f} | F1: {valid_f1:.4f} | Conf: {valid_conf:.4f}")
        
        # Save Best Model based on Validation Accuracy (can be changed to F1)
        if valid_acc > best_valid_acc and not args.dry_run:
            best_valid_acc = valid_acc
            patience_counter = 0
            
            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            unique_save_path = os.path.join(Config.MODEL_SAVE_DIR, f"{Config.SLIT_LAMP_MODEL_NAME}_best_{timestamp}.pth")
            standard_save_path = os.path.join(Config.MODEL_SAVE_DIR, f"{Config.SLIT_LAMP_MODEL_NAME}_best.pth")
            
            torch.save(model.state_dict(), unique_save_path)
            torch.save(model.state_dict(), standard_save_path)
            
            print(f"Model saved to {unique_save_path} and updated {standard_save_path}")
        elif not args.dry_run:
            patience_counter += 1
            
        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break
            
        if args.dry_run:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=Config.SLIT_LAMP_EPOCHS, help="Number of epochs")
    parser.add_argument("--dry-run", action="store_true", help="Run a single epoch for testing")
    args = parser.parse_args()
    main(args)
