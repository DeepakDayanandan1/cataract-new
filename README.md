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

## for processing commands

How to Run:

```bash
Fundus Binary Only:
python backend/tools/prepare_datasets.py --dataset binary
Fundus Multiclass Only:
bash
python backend/tools/prepare_datasets.py --dataset multiclass
Slit Lamp Only:
bash
python backend/tools/prepare_datasets.py --dataset slit_lamp
All Datasets (Default):
bash
python backend/tools/prepare_datasets.py --dataset all
Disable Augmentation (only split/preprocess):
bash
python backend/tools/prepare_datasets.py --dataset binary --no-augment
```

## slitlamp flow

1. Loading
   -- Image is loaded as RGB.
2. Preprocessing (Standardization)
   -- The pipeline.py applies the following steps specifically for slit_lamp (when use_green_channel=False):

-- Resize: Resized to 224x224 pixels.
-- Noise Reduction: Applies Gaussian Blur with a kernel size of (5, 5) to reduce high-frequency noise.
-- Color-Aware Enhancement (CLAHE):
-- Converts image from RGB to LAB color space.
-- Extracts the L (Lightness) channel.
-- Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel only (clipLimit=2.0, tileGridSize=(8,8)).
-- Merges channels back and converts to RGB. Rationale: This enhances contrast without distorting the color information.
-- Normalization: Pixel values are scaled to the range [0, 1] (divided by 255.0).
3. Splitting
-- The dataset is split into Train, Validation, and Test sets (default: 70% / 15% / 15%).
4. Offline Augmentation (Train Set Only)
-- If augmentation is enabled, the definitions from augmentations.py (get_train_transforms(image_type='slit_lamp')) are applied to generate 4 additional copies per image:

-- Random Horizontal Flip: 50% probability.
-- Random Rotation: +/- 15 degrees (limited to keep upright orientation).
-- Color Jitter: Randomizes Brightness (0.8x-1.2x) and Contrast (0.8x-1.2x).
-- Random Affine:
Scaling: Zoom in/out by +/- 10% (0.9x to 1.1x).
Translation: Shift by up to 5% vertically/horizontally.

### training commands

```bash
# Fundus Binary
python backend/ml/training/train_binary.py --dry-run
# Train for specific epochs (default is 50)
python backend/ml/training/train_binary.py --epochs 100

# Fundus Multiclass
python backend/ml/training/train_multiclass.py --dry-run
python backend/ml/training/train_multiclass.py --epochs 100

# Slit Lamp
python backend/ml/training/train_slit_lamp.py --dry-run
python backend/ml/training/train_slit_lamp.py --epochs 100
```
