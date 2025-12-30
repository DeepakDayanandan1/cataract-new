
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from preprocessing.dataset import CataractDataset
from models.densenet import get_model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, leave=False)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loop.set_description(f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main(args):
    Config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    train_dataset = CataractDataset(root_dir=Config.RAW_DATA_DIR, split='train')
    valid_dataset = CataractDataset(root_dir=Config.RAW_DATA_DIR, split='valid')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS
    )
    
    print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")
    
    if args.dry_run:
        print("Dry run mode enabled. Training for limited batches.")
        # Logic to limit batches could be here, or just run 1 epoch with break
    
    # 2. Model, Loss, Optimizer
    model = get_model(num_classes=Config.NUM_CLASSES, dropout_rate=Config.DROPOUT_RATE)
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    best_valid_acc = 0.0
    patience_counter = 0
    
    # 3. Training Loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}")
        
        # Save Best Model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            patience_counter = 0
            save_path = os.path.join(Config.MODEL_SAVE_DIR, f"{Config.MODEL_NAME}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        else:
            patience_counter += 1
            
        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break
            
        if args.dry_run:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS, help="Number of epochs")
    parser.add_argument("--dry_run", action="store_true", help="Run a single epoch for testing")
    args = parser.parse_args()
    main(args)
