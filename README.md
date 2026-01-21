# Cataract AI Diagnostic System

A deep learning-based application for detecting cataracts using both **Fundus** and **Slit-Lamp** eye images.

## Features

- **Dual-Model Architecture**:
  - **Fundus Analysis**: Binary classification (Normal vs. Cataract).
  - **Slit-Lamp Analysis**: Multi-class classification (Normal, Immature, Mature).
- **Preprocessing Pipeline**:
  - Green Channel Extraction
  - Denoising (Gaussian Blur)
  - Contrast Enhancement (CLAHE)
- **Interactive Web UI**: Built with Flask, featuring drag-and-drop uploads and visualization of preprocessing steps.
- **Deep Learning**: Powered by **DenseNet169**.

## Directory Structure

```
project_root/
├── backend/                   # All Python Code
│   ├── app/                   # Flask Application
│   ├── ml/                    # Machine Learning Logic (Models, Preprocessing, Training)
│   └── config.py              # Configuration
├── data/                      # Datasets
│   ├── raw/                   # Original Datasets (Fundus Binary, Multiclass, Slit-Lamp)
│   └── processed/             # Processed images
├── saved_models/              # Trained model weights
├── run_app.py                 # Application Entry Point
└── README.md                  # Project Information
```

## Installation

1. **Clone the repository** (if applicable).
2. **Install Dependencies**:
   ```bash
   pip install torch torchvision numpy opencv-python Flask Pillow tqdm sklearn
   ```
   _Note: Ensure you have a CUDA-enabled GPU for faster training, though CPU is supported._

## Usage

### 1. Running the Application

Start the Flask web server:

```bash
python run_app.py
```

Open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 2. Training the Models

The application requires trained models to function effectively.

**Train Fundus Binary Model:**

```bash
python backend/ml/training/train_binary.py --epochs 20
```

**Train Fundus Multiclass Model:**

```bash
python backend/ml/training/train_multiclass.py --epochs 20
```

**Train Slit-Lamp Model:**

```bash
python backend/ml/training/train_slit_lamp.py --epochs 20
```

_Models are saved to the `saved_models/` directory._

## How it Works

1. **Upload**: Select a Fundus or Slit-Lamp image in the respective section.
2. **Analysis**: The image goes through the preprocessing pipeline.
3. **Inference**: The processed image is passed to the specific DenseNet169 model.
4. **Result**: The UI displays the original image, preprocessing steps, diagnosis, and confidence score.

## Configuration

Adjust hyperparameters and paths in `backend/config.py`.
