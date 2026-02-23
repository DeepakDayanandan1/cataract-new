import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from backend.config import Config
from backend.ml.preprocessing.dataset import SlitLampDataset
from backend.ml.models.densenet import get_model
from backend.ml.preprocessing.augmentations import get_valid_transforms

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for images, labels in loader:
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
    
    print("\n--- Final Evaluation Results ---")
    print(f"Loss: {epoch_loss:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f} (Sensitivity)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Avg Confidence: {avg_conf:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=Config.SLIT_LAMP_CLASSES, zero_division=0))

def main():
    import builtins
    
    # Redirect print to a file to avoid Windows CMD encoding/buffer issues
    log_file = open("eval_results_clean.txt", "w", encoding="utf-8")
    
    def print(*args, **kwargs):
        kwargs["file"] = log_file
        builtins.print(*args, **kwargs)
        log_file.flush()
        
    Config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Validation Data
    processed_root = os.path.join(Config.PROCESSED_DATA_DIR, 'slitlamp')
    valid_dataset = SlitLampDataset(
        root_dir=processed_root,
        split='val',
        transform=get_valid_transforms(image_type='slit_lamp'),
        is_preprocessed=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=Config.SLIT_LAMP_BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS
    )
    
    print(f"Evaluating on {len(valid_dataset)} validation samples...\n")
    
    # 2. Load the Model
    model = get_model(num_classes=Config.SLIT_LAMP_NUM_CLASSES, dropout_rate=Config.SLIT_LAMP_DROPOUT_RATE)
    model = model.to(device)
    
    model_path = os.path.join(Config.MODEL_SAVE_DIR, f"{Config.SLIT_LAMP_MODEL_NAME}_best.pth")
    if not os.path.exists(model_path):
        print(f"Error: Could not find saved model at {model_path}")
        print("Please ensure you have trained the model first.")
        return
        
    print(f"Loading best model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    criterion = nn.CrossEntropyLoss()
    
    # 3. Evaluate
    evaluate(model, valid_loader, criterion, device)

if __name__ == "__main__":
    main()
