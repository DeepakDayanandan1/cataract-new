
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

def test_binary_accuracy():
    print("Testing Binary Accuracy Logic (train.py)...")
    batch_size = 4
    # Simulate output (N, 1) and label (N, 1)
    outputs = torch.tensor([[10.0], [-10.0], [0.1], [-0.1]]) # Sigmoid > 0.5 for +ve
    # Predictions: 1, 0, 1, 0
    
    labels = torch.tensor([[1.0], [0.0], [0.0], [1.0]]) # 1, 0, 0, 1
    # Matches: Yes, Yes, No, No -> 2 correct
    
    # Logic from train.py:
    predicted = (torch.sigmoid(outputs) > 0.5).float()
    
    # Check shape
    print(f"Outputs shape: {outputs.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Predicted shape: {predicted.shape}")
    
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {acc}")
    
    assert correct == 2
    assert acc == 0.5
    print("Binary Accuracy Logic Passed.\n")

    # TEST POTENTIAL BROADCASTING ERROR
    # If labels were (N,) instead of (N, 1)
    labels_flat = torch.tensor([1.0, 0.0, 0.0, 1.0])
    print(f"Testing Broadcasting issue with flat labels: {labels_flat.shape}")
    comparison = (predicted == labels_flat)
    print(f"Comparison shape: {comparison.shape}") # Should be (4, 4) if broadcasted!
    
    if comparison.shape == (batch_size, batch_size):
        print("ALERT: Broadcasting occurred! This would be a bug if labels weren't unsqueezed.")
    else:
        print("No broadcasting.")

def test_multiclass_accuracy():
    print("Testing Multiclass Accuracy Logic (train_multiclass.py)...")
    # Logic: preds = argmax, accuracy_score
    outputs = torch.tensor([
        [0.1, 0.9, 0.0], # Class 1
        [0.8, 0.1, 0.1], # Class 0
        [0.2, 0.3, 0.5]  # Class 2
    ])
    
    labels = torch.tensor([1, 0, 2]) # All correct
    
    preds = torch.argmax(outputs, dim=1)
    print(f"Preds: {preds}")
    
    acc = accuracy_score(labels.numpy(), preds.numpy())
    print(f"Accuracy: {acc}")
    
    assert acc == 1.0
    print("Multiclass Accuracy Logic Passed.\n")

if __name__ == "__main__":
    test_binary_accuracy()
    test_multiclass_accuracy()
