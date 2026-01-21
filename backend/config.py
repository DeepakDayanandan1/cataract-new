
import os

class Config:
    # Project Paths
    # backend/config.py -> backend -> project_root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    
    # Raw Data
    RAW_DATA_BINARY = os.path.join(DATA_DIR, 'raw', 'fundus_binary')
    RAW_DATA_MULTICLASS = os.path.join(DATA_DIR, 'raw', 'fundus_multiclass')
    RAW_DATA_SLIT_LAMP = os.path.join(DATA_DIR, 'raw', 'slit_lamp', 'slit-lamp')
    
    # Processed Data
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
    
    # Models
    MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
    
    # Uploads (for web app)
    # Using relative path for Flask to handle, or absolute
    UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'backend', 'app', 'static', 'uploads')
    
    # Data params
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_WORKERS = 0 # Set to 0 for Windows compatibility
    
    # Model params
    MODEL_NAME = 'densenet169'
    NUM_CLASSES = 1 # Binary classification
    DROPOUT_RATE = 0.5
    
    # Slit-Lamp Params
    SLIT_LAMP_CLASSES = ['normal', 'immature', 'mature']
    SLIT_LAMP_NUM_CLASSES = 3
    SLIT_LAMP_MODEL_NAME = 'densenet169_slit_lamp'
    
    # Training params
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    PATIENCE = 5 # Early stopping patience
    
    # Random Seed
    SEED = 42

    # Offline Augmentation
    TARGET_SIZE = (224, 224)
    AUGMENTATION_ENABLED = True
    AUGMENTATION_FACTOR = 4

    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(Config.SPLITS_DIR, exist_ok=True)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

if __name__ == "__main__":
    Config.ensure_dirs()
    print("Directories ensured.")
