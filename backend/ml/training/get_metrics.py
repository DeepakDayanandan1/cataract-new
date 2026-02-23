import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from backend.config import Config
from backend.ml.preprocessing.dataset import SlitLampDataset
from backend.ml.models.densenet import get_model
from backend.ml.preprocessing.augmentations import get_valid_transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_root = os.path.join(Config.PROCESSED_DATA_DIR, 'slitlamp')
    valid_dataset = SlitLampDataset(root_dir=processed_root, split='val', transform=get_valid_transforms('slit_lamp'), is_preprocessed=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.SLIT_LAMP_BATCH_SIZE, shuffle=False)
    
    model = get_model(num_classes=Config.SLIT_LAMP_NUM_CLASSES, dropout_rate=Config.SLIT_LAMP_DROPOUT_RATE).to(device)
    model.load_state_dict(torch.load(os.path.join(Config.MODEL_SAVE_DIR, f"{Config.SLIT_LAMP_MODEL_NAME}_best.pth"), map_location=device))
    model.eval()

    all_preds, all_labels, all_conf = [], [], []
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images.to(device))
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_conf.extend(conf.cpu().numpy())

    avg_conf = sum(all_conf) / len(all_conf) if all_conf else 0.0

    res = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average='macro', zero_division=0),
        "recall": recall_score(all_labels, all_preds, average='macro', zero_division=0),
        "f1": f1_score(all_labels, all_preds, average='macro', zero_division=0),
        "confidence": float(avg_conf)
    }
    
    with open("eval_metrics.json", "w") as f:
        json.dump(res, f, indent=4)

if __name__ == "__main__":
    main()
