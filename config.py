
import os

class Config:
    # Project Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(BASE_DIR, 'Cataract')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
    
    # Data params
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_WORKERS = 0 # Set to 0 for Windows compatibility
    
    # Model params
    MODEL_NAME = 'densenet169'
    NUM_CLASSES = 1 # Binary classification
    DROPOUT_RATE = 0.5
    
    # Training params
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    PATIENCE = 5 # Early stopping patience
    
    # Random Seed
    SEED = 42

    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(Config.SPLITS_DIR, exist_ok=True)

if __name__ == "__main__":
    Config.ensure_dirs()
    print("Directories ensured.")
