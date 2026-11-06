"""Пути и константы проекта."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = DATA_PROCESSED_DIR / "train.csv"
TEST_PATH = DATA_PROCESSED_DIR / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
TRAIN_SCORES_PATH = REPORTS_DIR / "train_scores.csv"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "pd-credit-scoring"

TARGET = "default"
RAW_TARGET = "default.payment.next.month"

RANDOM_STATE = 42
