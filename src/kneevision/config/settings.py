from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

IMAGE_SIZE = (224, 224)
NUM_KL_CLASSES = 5

CNN_MODEL_NAME = "densenet121"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DEVICE = "cuda"
