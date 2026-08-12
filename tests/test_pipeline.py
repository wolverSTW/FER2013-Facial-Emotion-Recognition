"""
FER2013 Pipeline Integration Test
==================================

This script checks whether:

    Dataset
        ↓
    DataLoader
        ↓
    EfficientNetV2-S
        ↓
    Model output

are working correctly together.

This test does NOT train the model.
"""


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import torch


# ============================================================
# 2. IMPORT PROJECT MODULES
# ============================================================

from src.config import DEVICE

from src.dataset import (
    load_datasets,
    create_train_validation_split,
    create_validation_dataset,
    create_dataloaders,
)

from src.model import create_model


# ============================================================
# 3. MAIN TEST
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FER2013 PIPELINE INTEGRATION TEST")
    print("=" * 70)

    # ========================================================
    # 3.1 DEVICE
    # ========================================================

    print()
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )
    else:
        print("GPU: Not available")

    # ========================================================
    # 3.2 LOAD DATASETS
    # ========================================================

    print()
    print("Loading datasets...")

    train_dataset, test_dataset = (
        load_datasets()
    )

    print(
        f"Original train dataset: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Test dataset: "
        f"{len(test_dataset):,}"
    )

    # ========================================================
    # 3.3 TRAIN / VALIDATION SPLIT
    # ========================================================

    print()
    print(
        "Creating train/validation split..."
    )

    train_subset, validation_subset = (
        create_train_validation_split(
            train_dataset
        )
    )

    # ========================================================
    # 3.4 VALIDATION DATASET
    # ========================================================

    validation_dataset = (
        create_validation_dataset(
            train_dataset,
            validation_subset,
        )
    )

    print(
        f"Training subset: "
        f"{len(train_subset):,}"
    )

    print(
        f"Validation dataset: "
        f"{len(validation_dataset):,}"
    )

    # ========================================================
    # 3.5 CREATE DATALOADERS
    # ========================================================

    print()
    print(
        "Creating DataLoaders..."
    )

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_dataloaders(
        train_subset,
        validation_dataset,
        test_dataset,
    )

    print(
        f"Training batches: "
        f"{len(train_loader):,}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader):,}"
    )

    print(
        f"Test batches: "
        f"{len(test_loader):,}"
    )

    # ========================================================
    # 3.6 GET ONE TRAINING BATCH
    # ========================================================

    print()
    print(
        "Reading one training batch..."
    )

    images, labels = next(
        iter(train_loader)
    )

    print(
        f"Images shape: "
        f"{images.shape}"
    )

    print(
        f"Labels shape: "
        f"{labels.shape}"
    )

    print(
        f"First 10 labels: "
        f"{labels[:10].tolist()}"
    )

    # ========================================================
    # 3.7 CHECK IMAGE SHAPE
    # ========================================================

    expected_image_shape = (
        images.shape[0],
        3,
        224,
        224,
    )

    if tuple(images.shape) != expected_image_shape:

        raise ValueError(
            "Unexpected image shape. "
            f"Expected {expected_image_shape}, "
            f"but received {tuple(images.shape)}."
        )

    print(
        "Image shape check: PASS"
    )

    # ========================================================
    # 3.8 CHECK LABEL SHAPE
    # ========================================================

    expected_label_shape = (
        images.shape[0],
    )

    if tuple(labels.shape) != expected_label_shape:

        raise ValueError(
            "Unexpected label shape. "
            f"Expected {expected_label_shape}, "
            f"but received {tuple(labels.shape)}."
        )

    print(
        "Label shape check: PASS"
    )

    # ========================================================
    # 3.9 CREATE MODEL
    # ========================================================

    print()
    print(
        "Creating EfficientNetV2-S model..."
    )

    model = create_model()

    model = model.to(
        DEVICE
    )

    model.eval()

    # ========================================================
    # 3.10 FORWARD PASS
    # ========================================================

    print()
    print(
        "Running forward pass..."
    )

    images = images.to(
        DEVICE
    )

    with torch.no_grad():

        outputs = model(
            images
        )

    # ========================================================
    # 3.11 DISPLAY OUTPUT
    # ========================================================

    print(
        f"Model output shape: "
        f"{outputs.shape}"
    )

    expected_output_shape = (
        images.shape[0],
        7,
    )

    print(
        f"Expected output shape: "
        f"{expected_output_shape}"
    )

    # ========================================================
    # 3.12 CHECK MODEL OUTPUT
    # ========================================================

    if tuple(outputs.shape) != expected_output_shape:

        raise ValueError(
            "Unexpected model output shape. "
            f"Expected {expected_output_shape}, "
            f"but received {tuple(outputs.shape)}."
        )

    print(
        "Model output shape check: PASS"
    )

    # ========================================================
    # 3.13 PREDICTION TEST
    # ========================================================

    predictions = outputs.argmax(
        dim=1
    )

    print()
    print(
        f"Predictions shape: "
        f"{predictions.shape}"
    )

    print(
        f"First 10 predictions: "
        f"{predictions[:10].tolist()}"
    )

    # ========================================================
    # 3.14 FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("PIPELINE INTEGRATION TEST PASSED")
    print("=" * 70)
    print()

    print(
        "Dataset       : PASS"
    )

    print(
        "DataLoader    : PASS"
    )

    print(
        "Image shape   : PASS"
    )

    print(
        "Label shape   : PASS"
    )

    print(
        "Model         : PASS"
    )

    print(
        "Forward pass  : PASS"
    )

    print()


# ============================================================
# 4. RUN TEST
# ============================================================

if __name__ == "__main__":
    main()