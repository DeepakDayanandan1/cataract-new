
import os
import sys
import uuid
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from models.densenet import get_model
from preprocessing.pipeline import preprocess_pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('ui', 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=Config.NUM_CLASSES, dropout_rate=0.0, pretrained=False)

# Load best weights if available, else standard
model_path = os.path.join(Config.MODEL_SAVE_DIR, f"{Config.MODEL_NAME}_best.pth")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")
else:
    print("Warning: Best model not found. Using variable weights (expect random predictions).")


model.to(device)
model.eval()

# Load Slit-Lamp Model
slit_lamp_model = get_model(num_classes=Config.SLIT_LAMP_NUM_CLASSES, dropout_rate=0.0, pretrained=False)
slit_lamp_model_path = os.path.join(Config.MODEL_SAVE_DIR, f"{Config.SLIT_LAMP_MODEL_NAME}_best.pth")
if os.path.exists(slit_lamp_model_path):
    slit_lamp_model.load_state_dict(torch.load(slit_lamp_model_path, map_location=device))
    print(f"Loaded slit-lamp model from {slit_lamp_model_path}")
else:
    print("Warning: Slit-lamp model not found. Using variable weights.")

slit_lamp_model.to(device)
slit_lamp_model.eval()

def save_temp_image(image_array, prefix="step"):
    """Saves a numpy image to static/uploads and returns the relative URL."""
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if grayscale/2D
    if len(image_array.shape) == 2:
        img = Image.fromarray((image_array).astype(np.uint8))
    else:
        # Convert RGB to BGR for PIL if needed, but pipeline returns RGB usually
        # Pipeline returns float 0-1 or uint8 0-255?
        # Let's check pipeline.
        # Pipeline:
        # resize -> cv2 (BGR/RGB?) -> Pipeline uses cv2, so likely BGR internally if not handled.
        # Wait, pipeline.py: load_image converts to RGB.
        # image_loader.py: load_image returns RGB.
        # pipeline.py: extract_green_channel returns 2D. 
        # CLAHE returns 2D.
        # Normalize returns float 0-1.
        
        # We need to handle float arrays [0, 1] -> [0, 255]
        if image_array.dtype == np.float32 or image_array.dtype == np.float64:
             if image_array.max() <= 1.0:
                 image_array = (image_array * 255).astype(np.uint8)
             else:
                 image_array = image_array.astype(np.uint8)
        
        img = Image.fromarray(image_array)

    img.save(filepath)
    return url_for('static', filename=f'uploads/{filename}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 2. Load and Preprocess
        # We need to reuse image_loader logic but file is in memory/disk
        from preprocessing.image_loader import load_image
        image = load_image(filepath) # Returns RGB
        
        # Get intermediate steps
        processed_input, steps = preprocess_pipeline(image, target_size=Config.IMAGE_SIZE)
        
        # 3. Save Steps for Visualization
        visualizations = {}
        visualizations['original'] = url_for('static', filename=f'uploads/{filename}')
        
        # Steps keys: 'resized', 'green_channel', 'denoised', 'enhanced'
        visualizations['green'] = save_temp_image(steps['green_channel'], 'green')
        visualizations['denoised'] = save_temp_image(steps['denoised'], 'denoised')
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

@app.route('/predict_slit_lamp', methods=['POST'])
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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 2. Load and Preprocess
        from preprocessing.image_loader import load_image
        image = load_image(filepath) # Returns RGB
        
        # Get intermediate steps
        processed_input, steps = preprocess_pipeline(image, target_size=Config.IMAGE_SIZE, use_green_channel=False)
        
        # 3. Save Steps for Visualization
        visualizations = {}
        visualizations['original'] = url_for('static', filename=f'uploads/{filename}')
        
        visualizations['denoised'] = save_temp_image(steps['denoised'], 'slit_denoised')
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
