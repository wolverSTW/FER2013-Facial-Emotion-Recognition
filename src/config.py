"""
FER2013 Facial Emotion Recognition
==================================

This module stores all configurable parameters used throughout
the Facial Emotion Recognition project.

Keeping configuration values in one place makes the project
easier to maintain, reproduce, and modify.

The configuration supports:

    - Local CPU development
    - NVIDIA CUDA GPU training
    - Google Colab
    - Vast.ai GPU servers
"""


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from pathlib import Path

import torch


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# Project root:
#
# FER2013-Facial-Emotion-Recognition/
#
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# ============================================================
# 2.1 DATASET DIRECTORIES
# ============================================================

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "FER2013"
)

TRAIN_DIR = (
    DATA_DIR
    / "train"
)

TEST_DIR = (
    DATA_DIR
    / "test"
)


# ============================================================
# 2.2 OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

MODEL_DIR = (
    OUTPUT_DIR
    / "models"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

TABLE_DIR = (
    OUTPUT_DIR
    / "tables"
)

METRIC_DIR = (
    OUTPUT_DIR
    / "metrics"
)

LOG_DIR = (
    OUTPUT_DIR
    / "logs"
)

RESULTS_DIR = (
    OUTPUT_DIR
    / "results"
)


# ============================================================
# 3. DATASET CONFIGURATION
# ============================================================

# FER2013 contains seven emotion classes.

NUM_CLASSES = 7


# Emotion class names.
#
# IMPORTANT:
# These names must match the folder names in:
#
# data/FER2013/train/
# data/FER2013/test/
#

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


# ============================================================
# 4. IMAGE CONFIGURATION
# ============================================================

# Original FER2013 images are 48 x 48 pixels.
#
# EfficientNetV2-S works with larger input images,
# therefore the images are resized to 224 x 224.

IMAGE_SIZE = 224


# Number of images processed in one training step.

BATCH_SIZE = 64


# Number of CPU worker processes used by DataLoader.

NUM_WORKERS = 4


# ============================================================
# 5. TRAINING CONFIGURATION
# ============================================================

# Maximum number of training epochs.

NUM_EPOCHS = 30


# Initial learning rate.

LEARNING_RATE = 1e-4


# AdamW weight decay.

WEIGHT_DECAY = 1e-4


# Random seed for reproducibility.

RANDOM_SEED = 42


# Early stopping patience.
#
# Training stops if validation accuracy does not improve
# for this number of consecutive epochs.

PATIENCE = 5


# ============================================================
# 6. DATA SPLIT CONFIGURATION
# ============================================================

# Percentage of the original training dataset used
# for model training.

TRAIN_SPLIT = 0.90


# Percentage of the original training dataset used
# for validation.

VALIDATION_SPLIT = 0.10


# ============================================================
# 7. DEVICE CONFIGURATION
# ============================================================

# Automatically use NVIDIA CUDA when available.
#
# Otherwise use CPU.

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 7.1 CUDA INFORMATION
# ============================================================

# True when an NVIDIA CUDA GPU is available.

USE_CUDA = (
    torch.cuda.is_available()
)


# GPU name.
#
# This is useful for experiment logs and reports.

GPU_NAME = (
    torch.cuda.get_device_name(0)
    if USE_CUDA
    else "CPU"
)


# ============================================================
# 8. IMAGENET NORMALIZATION
# ============================================================

# EfficientNetV2-S uses ImageNet pretrained weights.
#
# Therefore ImageNet normalization is applied.

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# ============================================================
# 9. PRINT CONFIGURATION
# ============================================================

def print_config() -> None:
    """
    Print the main project configuration.

    This function is useful for checking the configuration
    before starting model training.
    """

    print()

    print("=" * 70)

    print(
        "FER2013 PROJECT CONFIGURATION"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Project information
    # --------------------------------------------------------

    print()

    print("Project")

    print("-" * 70)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Dataset      : {DATA_DIR}"
    )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print()

    print("Dataset")

    print("-" * 70)

    print(
        f"Classes      : {NUM_CLASSES}"
    )

    print(
        f"Class names  : {CLASS_NAMES}"
    )

    # --------------------------------------------------------
    # Image information
    # --------------------------------------------------------

    print()

    print("Image")

    print("-" * 70)

    print(
        f"Image size   : "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print(
        f"Batch size   : {BATCH_SIZE}"
    )

    print(
        f"Workers      : {NUM_WORKERS}"
    )

    # --------------------------------------------------------
    # Training information
    # --------------------------------------------------------

    print()

    print("Training")

    print("-" * 70)

    print(
        f"Epochs       : {NUM_EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Weight decay : {WEIGHT_DECAY}"
    )

    print(
        f"Patience     : {PATIENCE}"
    )

    print(
        f"Random seed  : {RANDOM_SEED}"
    )

    # --------------------------------------------------------
    # Data split
    # --------------------------------------------------------

    print()

    print("Data Split")

    print("-" * 70)

    print(
        f"Train split  : "
        f"{TRAIN_SPLIT:.0%}"
    )

    print(
        f"Validation   : "
        f"{VALIDATION_SPLIT:.0%}"
    )

    # --------------------------------------------------------
    # Device information
    # --------------------------------------------------------

    print()

    print("Device")

    print("-" * 70)

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"CUDA enabled : {USE_CUDA}"
    )

    print(
        f"GPU          : {GPU_NAME}"
    )

    # --------------------------------------------------------
    # End
    # --------------------------------------------------------

    print()

    print("=" * 70)