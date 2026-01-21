
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from backend.config import Config
from backend.ml.preprocessing.dataset import SlitLampDataset
from backend.ml.models.densenet import get_model
from backend.ml.preprocessing.augmentations import get_train_transforms, get_valid_transforms
from torch.utils.data import Dataset

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, leave=False)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device).long() # CrossEntropyLoss expects long labels
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
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
            labels = labels.to(device).long()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
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
    # 1. Prepare Data
    full_dataset = SlitLampDataset(root_dir=Config.RAW_DATA_SLIT_LAMP)
    
    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_subset, valid_subset = random_split(full_dataset, [train_size, valid_size])
    
    class TransformDataset(Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
            
        def __len__(self):
            return len(self.subset)

    train_dataset = TransformDataset(train_subset, transform=get_train_transforms('slit_lamp'))
    valid_dataset = TransformDataset(valid_subset, transform=get_valid_transforms('slit_lamp'))
    
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
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")
    
    if args.dry_run:
        print("Dry run mode enabled.")
    
    # 2. Model, Loss, Optimizer
    # Config.SLIT_LAMP_NUM_CLASSES is 3
    model = get_model(num_classes=Config.SLIT_LAMP_NUM_CLASSES, dropout_rate=Config.DROPOUT_RATE)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
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
            save_path = os.path.join(Config.MODEL_SAVE_DIR, f"{Config.SLIT_LAMP_MODEL_NAME}_best.pth")
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
