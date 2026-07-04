"""Project-wide configuration for the bike sharing LSTM pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"

TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
MODEL_PATH = MODELS_DIR / "lstm_model.keras"
H5_MODEL_PATH = MODELS_DIR / "lstm_model.h5"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
HISTORY_PATH = LOGS_DIR / "training_history.csv"
TRAINING_PLOT_PATH = LOGS_DIR / "training_curve.png"
PREDICTION_PLOT_PATH = OUTPUT_DIR / "prediction_visualization.png"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

LOOKBACK = 24
BATCH_SIZE = 64
EPOCHS = 100
LSTM_UNITS = [64, 32]
DENSE_UNITS = 16
DROPOUT = 0.2
LEARNING_RATE = 0.001
PATIENCE = 8
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
PREDICTION_PLOT_TAIL = 24 * 7



def ensure_project_dirs() -> None:
    """Create project output directories when they do not exist."""
    for directory in (DATA_DIR, MODELS_DIR, LOGS_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
