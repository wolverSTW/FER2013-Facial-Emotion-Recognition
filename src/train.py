"""
FER2013 Facial Emotion Recognition
==================================

This module implements the complete training pipeline for
the FER2013 facial emotion recognition project.

Main components:
    - EfficientNetV2-S model
    - Cross-Entropy Loss
    - AdamW optimizer
    - Cosine Annealing learning-rate scheduler
    - Automatic Mixed Precision (AMP) for CUDA GPUs
    - Best-model checkpointing
    - Early stopping
    - Training history recording
    - JSON result saving

Dataset split:
    - 90% of the original training data -> Training
    - 10% of the original training data -> Validation
    - Official FER2013 test set -> Final testing

The code is designed to run on:
    - CPU
    - NVIDIA CUDA-enabled GPU

Author:
    FER2013 Facial Emotion Recognition Project
"""


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import json

from typing import Dict, Tuple

import torch

from torch import nn

from torch.optim import AdamW

from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.amp import GradScaler, autocast


# ============================================================
# 2. IMPORT PROJECT MODULES
# ============================================================

from src.config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    MODEL_DIR,
)

from src.dataset import (
    load_datasets,
    validate_class_mapping,
    create_train_validation_split,
    create_validation_dataset,
    create_dataloaders,
)

from src.model import create_model


# ============================================================
# 3. PREPARE OUTPUT DIRECTORIES
# ============================================================

def prepare_output_directories() -> None:
    """
    Create the required output directories.

    The project stores:
        - trained models in outputs/models/
        - training results in outputs/results/
    """

    # --------------------------------------------------------
    # Create model directory.
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Create results directory.
    # --------------------------------------------------------

    results_dir = MODEL_DIR.parent / "results"

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 4. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    scaler: GradScaler,
) -> Tuple[float, float]:
    """
    Train the model for one complete epoch.

    Parameters
    ----------
    model : nn.Module
        Neural network model.

    dataloader : DataLoader
        Training DataLoader.

    criterion : nn.Module
        Loss function.

    optimizer
        Optimizer used to update model parameters.

    device : torch.device
        CPU or CUDA device.

    scaler : GradScaler
        Gradient scaler used for mixed-precision training.

    Returns
    -------
    Tuple[float, float]
        Average training loss and training accuracy.
    """

    # --------------------------------------------------------
    # Set model to training mode.
    # --------------------------------------------------------

    model.train()

    # --------------------------------------------------------
    # Variables for calculating statistics.
    # --------------------------------------------------------

    running_loss = 0.0

    correct_predictions = 0

    total_samples = 0

    # --------------------------------------------------------
    # Process each batch.
    # --------------------------------------------------------

    for images, labels in dataloader:

        # ----------------------------------------------------
        # Move images to CPU or GPU.
        # ----------------------------------------------------

        images = images.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Move labels to CPU or GPU.
        # ----------------------------------------------------

        labels = labels.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Clear previous gradients.
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Automatic Mixed Precision.
        #
        # AMP is enabled only when CUDA is available.
        # ----------------------------------------------------

        with autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):

            # Forward pass.
            outputs = model(images)

            # Calculate classification loss.
            loss = criterion(
                outputs,
                labels,
            )

        # ----------------------------------------------------
        # Backpropagation.
        # ----------------------------------------------------

        scaler.scale(loss).backward()

        # ----------------------------------------------------
        # Update model parameters.
        # ----------------------------------------------------

        scaler.step(optimizer)

        # ----------------------------------------------------
        # Update gradient scaler.
        # ----------------------------------------------------

        scaler.update()

        # ----------------------------------------------------
        # Accumulate loss.
        # ----------------------------------------------------

        running_loss += (
            loss.item()
            * images.size(0)
        )

        # ----------------------------------------------------
        # Get predicted class.
        #
        # outputs shape:
        #     [batch_size, 7]
        #
        # argmax(dim=1):
        #     predicted emotion class
        # ----------------------------------------------------

        predictions = outputs.argmax(
            dim=1
        )

        # ----------------------------------------------------
        # Count correct predictions.
        # ----------------------------------------------------

        correct_predictions += (
            (predictions == labels)
            .sum()
            .item()
        )

        # ----------------------------------------------------
        # Count total samples.
        # ----------------------------------------------------

        total_samples += labels.size(0)

    # ========================================================
    # Calculate epoch statistics.
    # ========================================================

    epoch_loss = (
        running_loss
        / total_samples
    )

    epoch_accuracy = (
        correct_predictions
        / total_samples
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


# ============================================================
# 5. VALIDATION
# ============================================================

@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation dataset.

    Gradients are disabled because model parameters are not
    updated during validation.

    Parameters
    ----------
    model : nn.Module
        Neural network model.

    dataloader : DataLoader
        Validation DataLoader.

    criterion : nn.Module
        Loss function.

    device : torch.device
        CPU or CUDA device.

    Returns
    -------
    Tuple[float, float]
        Average validation loss and validation accuracy.
    """

    # --------------------------------------------------------
    # Set model to evaluation mode.
    # --------------------------------------------------------

    model.eval()

    # --------------------------------------------------------
    # Variables for validation statistics.
    # --------------------------------------------------------

    running_loss = 0.0

    correct_predictions = 0

    total_samples = 0

    # --------------------------------------------------------
    # Process each validation batch.
    # --------------------------------------------------------

    for images, labels in dataloader:

        # ----------------------------------------------------
        # Move images to device.
        # ----------------------------------------------------

        images = images.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Move labels to device.
        # ----------------------------------------------------

        labels = labels.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Automatic Mixed Precision for CUDA.
        # ----------------------------------------------------

        with autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):

            # Forward pass.
            outputs = model(images)

            # Calculate validation loss.
            loss = criterion(
                outputs,
                labels,
            )

        # ----------------------------------------------------
        # Accumulate validation loss.
        # ----------------------------------------------------

        running_loss += (
            loss.item()
            * images.size(0)
        )

        # ----------------------------------------------------
        # Get predictions.
        # ----------------------------------------------------

        predictions = outputs.argmax(
            dim=1
        )

        # ----------------------------------------------------
        # Count correct predictions.
        # ----------------------------------------------------

        correct_predictions += (
            (predictions == labels)
            .sum()
            .item()
        )

        # ----------------------------------------------------
        # Count total samples.
        # ----------------------------------------------------

        total_samples += labels.size(0)

    # ========================================================
    # Calculate validation statistics.
    # ========================================================

    epoch_loss = (
        running_loss
        / total_samples
    )

    epoch_accuracy = (
        correct_predictions
        / total_samples
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


# ============================================================
# 6. SAVE TRAINING HISTORY
# ============================================================

def save_training_history(
    history: Dict[str, list],
) -> None:
    """
    Save training history to a JSON file.

    The saved file can later be used to generate:

        - Training accuracy plot
        - Validation accuracy plot
        - Training loss plot
        - Validation loss plot
        - Learning-rate plot

    Parameters
    ----------
    history : Dict[str, list]
        Training history dictionary.
    """

    # --------------------------------------------------------
    # Define results directory.
    # --------------------------------------------------------

    results_dir = (
        MODEL_DIR.parent
        / "results"
    )

    # --------------------------------------------------------
    # Ensure directory exists.
    # --------------------------------------------------------

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Define output path.
    # --------------------------------------------------------

    history_path = (
        results_dir
        / "training_history.json"
    )

    # --------------------------------------------------------
    # Save JSON file.
    # --------------------------------------------------------

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
        )

    print()

    print(
        "Training history saved to:"
    )

    print(
        history_path
    )


# ============================================================
# 7. TRAIN MODEL
# ============================================================

def train_model(
    model: nn.Module,
    train_loader,
    validation_loader,
    num_epochs: int = NUM_EPOCHS,
) -> Dict[str, list]:
    """
    Train the model and validate after every epoch.

    Training components:
        - CrossEntropyLoss
        - AdamW
        - CosineAnnealingLR
        - Automatic Mixed Precision
        - Best-model checkpointing
        - Early stopping

    Parameters
    ----------
    model : nn.Module
        Neural network model.

    train_loader : DataLoader
        Training DataLoader.

    validation_loader : DataLoader
        Validation DataLoader.

    num_epochs : int
        Maximum number of epochs.

    Returns
    -------
    Dict[str, list]
        Training history.
    """

    # ========================================================
    # 7.1 LOSS FUNCTION
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # 7.2 OPTIMIZER
    # ========================================================

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # 7.3 LEARNING-RATE SCHEDULER
    # ========================================================

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
    )

    # ========================================================
    # 7.4 MIXED-PRECISION SCALER
    # ========================================================

    scaler = GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda"),
    )

    # ========================================================
    # 7.5 TRAINING HISTORY
    # ========================================================

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    # ========================================================
    # 7.6 BEST MODEL TRACKING
    # ========================================================

    best_val_accuracy = 0.0

    epochs_without_improvement = 0

    # ========================================================
    # 7.7 EPOCH LOOP
    # ========================================================

    for epoch in range(
        num_epochs
    ):

        print()

        print(
            "=" * 70
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs}"
        )

        print(
            "=" * 70
        )

        # ====================================================
        # TRAINING
        # ====================================================

        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=DEVICE,
                scaler=scaler,
            )
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        val_loss, val_accuracy = (
            validate_one_epoch(
                model=model,
                dataloader=validation_loader,
                criterion=criterion,
                device=DEVICE,
            )
        )

        # ====================================================
        # UPDATE LEARNING RATE
        # ====================================================

        scheduler.step()

        # ----------------------------------------------------
        # Get current learning rate.
        # ----------------------------------------------------

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ====================================================
        # SAVE TRAINING HISTORY
        # ====================================================

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "train_accuracy"
        ].append(
            train_accuracy
        )

        history[
            "val_loss"
        ].append(
            val_loss
        )

        history[
            "val_accuracy"
        ].append(
            val_accuracy
        )

        history[
            "learning_rate"
        ].append(
            current_lr
        )

        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        print(
            f"Train Loss      : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy  : "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation Loss : "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation Acc. : "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Learning Rate   : "
            f"{current_lr:.8f}"
        )

        # ====================================================
        # CHECK FOR BEST MODEL
        # ====================================================

        if (
            val_accuracy
            > best_val_accuracy
        ):

            # ------------------------------------------------
            # Update best accuracy.
            # ------------------------------------------------

            best_val_accuracy = (
                val_accuracy
            )

            # ------------------------------------------------
            # Reset early stopping counter.
            # ------------------------------------------------

            epochs_without_improvement = 0

            # ------------------------------------------------
            # Define model path.
            # ------------------------------------------------

            best_model_path = (
                MODEL_DIR
                / "best_model.pth"
            )

            # ------------------------------------------------
            # Save model weights.
            # ------------------------------------------------

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            print()

            print(
                "New best model saved!"
            )

            print(
                f"Best Validation Accuracy: "
                f"{best_val_accuracy:.4f}"
            )

            print(
                f"Model path: "
                f"{best_model_path}"
            )

        # ====================================================
        # NO IMPROVEMENT
        # ====================================================

        else:

            epochs_without_improvement += 1

            print()

            print(
                "No validation improvement "
                f"for {epochs_without_improvement} "
                "epoch(s)."
            )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()

            print(
                "Early stopping triggered."
            )

            print(
                f"Training stopped after "
                f"{epoch + 1} epochs."
            )

            break

    # ========================================================
    # TRAINING COMPLETED
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "TRAINING COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Save training history.
    # --------------------------------------------------------

    save_training_history(
        history
    )

    return history


# ============================================================
# 8. MAIN TRAINING PIPELINE
# ============================================================

def main() -> None:
    """
    Execute the complete FER2013 training pipeline.

    Pipeline:

        1. Prepare output directories
        2. Load datasets
        3. Validate class mapping
        4. Split training data
        5. Create validation dataset
        6. Create DataLoaders
        7. Create EfficientNetV2-S model
        8. Move model to CPU/GPU
        9. Train model
        10. Save best model
        11. Save training history

    Note:
        The official FER2013 test dataset is NOT used during
        training. It will be used later by evaluate.py for
        final model evaluation.
    """

    # ========================================================
    # 8.1 PREPARE OUTPUT DIRECTORIES
    # ========================================================

    prepare_output_directories()

    # ========================================================
    # 8.2 DISPLAY DEVICE INFORMATION
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "FER2013 FACIAL EMOTION RECOGNITION"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Using device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Display GPU information.
    # --------------------------------------------------------

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA Version: "
            f"{torch.version.cuda}"
        )

    else:

        print(
            "GPU: Not available"
        )

        print(
            "Training will run on CPU."
        )

    print()

    # ========================================================
    # 8.3 LOAD DATASETS
    # ========================================================

    print(
        "Loading FER2013 datasets..."
    )

    train_dataset, test_dataset = (
        load_datasets()
    )

    # ========================================================
    # 8.4 VALIDATE CLASS MAPPING
    # ========================================================

    print(
        "Validating class mapping..."
    )

    validate_class_mapping(
        train_dataset,
        test_dataset,
    )

    print(
        "Class mapping: OK"
    )

    # ========================================================
    # 8.5 DISPLAY ORIGINAL DATASET SIZE
    # ========================================================

    print()

    print(
        f"Original training samples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Official test samples: "
        f"{len(test_dataset):,}"
    )

    # ========================================================
    # 8.6 CREATE TRAIN / VALIDATION SPLIT
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
    # 8.7 CREATE VALIDATION DATASET
    # ========================================================

    validation_dataset = (
        create_validation_dataset(
            train_dataset,
            validation_subset,
        )
    )

    # ========================================================
    # 8.8 DISPLAY SPLIT INFORMATION
    # ========================================================

    print()

    print(
        f"Training samples: "
        f"{len(train_subset):,}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset):,}"
    )

    print(
        f"Test samples: "
        f"{len(test_dataset):,}"
    )

    # ========================================================
    # 8.9 CREATE DATALOADERS
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

    print()

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
    # 8.10 CREATE MODEL
    # ========================================================

    print()

    print(
        "Creating EfficientNetV2-S model..."
    )

    model = create_model()

    # ========================================================
    # 8.11 MOVE MODEL TO DEVICE
    # ========================================================

    model = model.to(
        DEVICE
    )

    # ========================================================
    # 8.12 COUNT MODEL PARAMETERS
    # ========================================================

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # ========================================================
    # 8.13 START TRAINING
    # ========================================================

    print()

    print(
        "Starting training..."
    )

    print()

    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        num_epochs=NUM_EPOCHS,
    )

    # ========================================================
    # 8.14 FINAL MESSAGE
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "ALL TRAINING TASKS COMPLETED"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Saved files:"
    )

    print(
        f"Best model: "
        f"{MODEL_DIR / 'best_model.pth'}"
    )

    print(
        f"Training history: "
        f"{MODEL_DIR.parent / 'results' / 'training_history.json'}"
    )

    print()

    print(
        "The official test set has NOT been used "
        "during training."
    )

    print(
        "It will be evaluated later using evaluate.py."
    )

    print()

    return history


# ============================================================
# 9. RUN MAIN
# ============================================================

if __name__ == "__main__":
    main()