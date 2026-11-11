
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "final"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

# Model
MODEL_NAME = "distilbert-base-uncased"

LABELS = [
    "anger", "disgust", "fear", "joy",
    "sadness", "surprise", "neutral",
]

NUM_LABELS = len(LABELS)

LABEL2ID = {
    label: i for i, label in enumerate(LABELS)
}

ID2LABEL = {
    i: label for i, label in enumerate(LABELS)
}

# General settings
SEED = 42
DEFAULT_THRESHOLD = 0.5
MIN_POSITIVE_PER_LABEL = 500

# Baseline defaults
MAX_LENGTH = 64
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

# Controlled experiments:
# Change one hyperparameter at a time from E1.
EXPERIMENTS = {
    "E1": {
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3,
        "max_length": 64,
    },
    "E2": {
        "learning_rate": 3e-5,
        "batch_size": 16,
        "epochs": 3,
        "max_length": 64,
    },
    "E3": {
        "learning_rate": 5e-5,
        "batch_size": 16,
        "epochs": 3,
        "max_length": 64,
    },
    "E4": {
        "learning_rate": 2e-5,
        "batch_size": 32,
        "epochs": 3,
        "max_length": 64,
    },
    "E5": {
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 5,
        "max_length": 64,
    },
    "E6": {
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3,
        "max_length": 128,
    },
}
