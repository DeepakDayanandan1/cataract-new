
import os
import sys
import uuid
import torch
import numpy as np
from flask import Blueprint, render_template, request, jsonify, url_for, current_app
from PIL import Image

# Import from backend packages
from backend.config import Config
from backend.ml.models.densenet import get_model
from backend.ml.preprocessing.pipeline import preprocess_pipeline
from backend.ml.preprocessing.image_loader import load_image

# Create Blueprint
main = Blueprint('main', __name__)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models (Load lazily or globally? Globally is fine for single worker)
# We need to make sure Config paths are correct
try:
    model = get_model(num_classes=Config.NUM_CLASSES, dropout_rate=0.0, pretrained=False)
    model_path = os.path.join(Config.MODEL_SAVE_DIR, f"{Config.MODEL_NAME}_best.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Best model not found at {model_path}. Using variable weights.")
    model.to(device)
except Exception as e:
    print(f"Error loading binary model: {e}")

try:
    multiclass_model = get_model(num_classes=4, dropout_rate=0.0, pretrained=False)
    multiclass_model_path = os.path.join(Config.MODEL_SAVE_DIR, "densenet_multiclass_best.pth")
    if os.path.exists(multiclass_model_path):
        multiclass_model.load_state_dict(torch.load(multiclass_model_path, map_location=device))
        print(f"Loaded multi-class model from {multiclass_model_path}")
    else:
        print(f"Warning: Multi-class model not found at {multiclass_model_path}. Using variable weights.")
    multiclass_model.to(device)
    multiclass_model.eval()
except Exception as e:
    print(f"Error loading multiclass model: {e}")

try:
    slit_lamp_model = get_model(num_classes=Config.SLIT_LAMP_NUM_CLASSES, dropout_rate=0.0, pretrained=False)
    slit_lamp_model_path = os.path.join(Config.MODEL_SAVE_DIR, f"{Config.SLIT_LAMP_MODEL_NAME}_best.pth")
    if os.path.exists(slit_lamp_model_path):
        slit_lamp_model.load_state_dict(torch.load(slit_lamp_model_path, map_location=device))
        print(f"Loaded slit-lamp model from {slit_lamp_model_path}")
    else:
        print(f"Warning: Slit-lamp model not found at {slit_lamp_model_path}. Using variable weights.")
    slit_lamp_model.to(device)
    slit_lamp_model.eval()
except Exception as e:
    print(f"Error loading slit-lamp model: {e}")

def save_temp_image(image_array, prefix="step"):
    """Saves a numpy image to static/uploads and returns the relative URL."""
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    
    # Check if grayscale/2D
    if len(image_array.shape) == 2:
        img = Image.fromarray((image_array).astype(np.uint8))
    else:
        if image_array.dtype == np.float32 or image_array.dtype == np.float64:
             if image_array.max() <= 1.0:
                 image_array = (image_array * 255).astype(np.uint8)
             else:
                 image_array = image_array.astype(np.uint8)
        
        img = Image.fromarray(image_array)

    img.save(filepath)
    # Return relative path for url_for('static', filename=...)
    # The blueprint is serving static files from 'backend/app/static', mapped to /static
    # Filename should be 'uploads/filename'
    return url_for('static', filename=f'uploads/{filename}')

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # 1. Save Original
        ext = os.path.splitext(file.filename)[1]
        filename = f"original_{uuid.uuid4().hex[:8]}{ext}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 2. Load and Preprocess
        # Reuse image_loader logic
        image = load_image(filepath) # Returns RGB
        
        # Get intermediate steps
        processed_input, steps = preprocess_pipeline(image, target_size=Config.IMAGE_SIZE)
        
        # 3. Save Steps for Visualization
        visualizations = {}
        visualizations['original'] = url_for('static', filename=f'uploads/{filename}')
        
        # Steps keys: 'resized', 'green_channel', 'denoised', 'enhanced'
        if 'green_channel' in steps:
            visualizations['green'] = save_temp_image(steps['green_channel'], 'green')
        if 'denoised' in steps:
            visualizations['denoised'] = save_temp_image(steps['denoised'], 'denoised')
        if 'enhanced' in steps:
            visualizations['clahe'] = save_temp_image(steps['enhanced'], 'clahe')
        
        # 4. Inference
        tensor_img = torch.tensor(processed_input).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor_img)
            probability = torch.sigmoid(output).item()
            
        prediction = "Cataract" if probability > 0.5 else "Normal"
        confidence = probability if probability > 0.5 else 1 - probability
        
        return jsonify({
            'prediction': prediction,
            'confidence': f"{confidence*100:.2f}%",
            'visualizations': visualizations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/predict_multiclass', methods=['POST'])
def predict_multiclass():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # 1. Save Original
        ext = os.path.splitext(file.filename)[1]
        filename = f"multi_original_{uuid.uuid4().hex[:8]}{ext}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 2. Load and Preprocess
        image = load_image(filepath) # Returns RGB
        
        # Use same pipeline as binary for consistency (Fundus)
        processed_input, steps = preprocess_pipeline(image, target_size=Config.IMAGE_SIZE)
        
        # 3. Save Steps for Visualization
        visualizations = {}
        visualizations['original'] = url_for('static', filename=f'uploads/{filename}')
        if 'green_channel' in steps:
            visualizations['green'] = save_temp_image(steps['green_channel'], 'multi_green')
        if 'denoised' in steps:
            visualizations['denoised'] = save_temp_image(steps['denoised'], 'multi_denoised')
        if 'enhanced' in steps:
            visualizations['clahe'] = save_temp_image(steps['enhanced'], 'multi_clahe')
        
        # 4. Inference
        tensor_img = torch.tensor(processed_input).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = multiclass_model(tensor_img)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
            
        classes = ['Normal', 'Mild', 'Moderate', 'Severe']
        prediction = classes[predicted_idx]
        
        # Model Performance Metrics (from Test Set)
        # TODO: Load these dynamically or from config if possible
        metrics = {
            'accuracy': '76.67%',
            'precision': '78.65%',
            'recall': '76.67%',
            'f1': '76.03%'
        }
        
        return jsonify({
            'prediction': prediction,
            'confidence': f"{confidence*100:.2f}%", 
            'metrics': metrics,
            'visualizations': visualizations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/predict_slit_lamp', methods=['POST'])
def predict_slit_lamp():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # 1. Save Original
        ext = os.path.splitext(file.filename)[1]
        filename = f"slit_original_{uuid.uuid4().hex[:8]}{ext}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 2. Load and Preprocess
        image = load_image(filepath) # Returns RGB
        
        # Get intermediate steps
        processed_input, steps = preprocess_pipeline(image, target_size=Config.IMAGE_SIZE, use_green_channel=False)
        
        # 3. Save Steps for Visualization
        visualizations = {}
        visualizations['original'] = url_for('static', filename=f'uploads/{filename}')
        
        if 'denoised' in steps:
            visualizations['denoised'] = save_temp_image(steps['denoised'], 'slit_denoised')
        if 'enhanced' in steps:
            visualizations['clahe'] = save_temp_image(steps['enhanced'], 'slit_clahe')
        
        # 4. Inference
        tensor_img = torch.tensor(processed_input).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = slit_lamp_model(tensor_img)
            # Softmax for multiclass
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
            
        prediction = Config.SLIT_LAMP_CLASSES[predicted_idx].capitalize()
        
        return jsonify({
            'prediction': prediction,
            'confidence': f"{confidence*100:.2f}%",
            'visualizations': visualizations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
