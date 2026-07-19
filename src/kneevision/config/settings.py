import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

IMAGE_SIZE = 224
NUM_KL_CLASSES = 5

CNN_MODEL_NAME = "densenet121"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
DEVICE = "cuda"
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.4
SAMPLER_POWER = 0.5
TTA_AUGS = 2
EARLY_STOP_PATIENCE = 15
CHECKPOINT_INTERVAL = 5
MODELS_DIR = PROJECT_ROOT / "models"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///" + str(PROJECT_ROOT / "mlflow.db"))
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "kneevision")
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "true").lower() == "true"

DVC_REMOTE = os.getenv("DVC_REMOTE", "storage")
DVC_REMOTE_URL = os.getenv("DVC_REMOTE_URL", "/tmp/dvc-storage")
