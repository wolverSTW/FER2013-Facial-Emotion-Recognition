"""
Project Configuration
=====================

This module stores all configurable parameters used throughout
the Facial Emotion Recognition project.

Keeping configuration values in one place makes the project
easier to maintain and modify.
"""

from pathlib import Path
import torch

# ==========================================================
# Project Paths
# ==========================================================

# Project root directory (FER2013-Facial-Emotion-Recognition/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset directory
DATA_DIR = PROJECT_ROOT / "data" / "FER2013"

# Training dataset directory
TRAIN_DIR = DATA_DIR / "train"

# Testing dataset directory
TEST_DIR = DATA_DIR / "test"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR = OUTPUT_DIR / "models"

FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR = OUTPUT_DIR / "tables"

METRIC_DIR = OUTPUT_DIR / "metrics"

LOG_DIR = OUTPUT_DIR / "logs"

# ==========================================================
# Dataset Configuration
# ==========================================================

# Number of emotion classes
NUM_CLASSES = 7

# Emotion labels
CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# ==========================================================
# Image Configuration
# ==========================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 2

# ==========================================================
# Training Configuration
# ==========================================================

NUM_EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42

# ==========================================================
# Device Configuration
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================================
# ImageNet Normalization
# ==========================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]