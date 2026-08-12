"""
Model Training Utilities
========================

This module contains the training and validation logic for the
FER2013 facial emotion recognition model.

The training pipeline supports both CPU and CUDA-enabled GPU
environments.
"""

from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MODEL_DIR,
)

from src.dataset import (
    load_datasets,
    validate_class_mapping,
    create_dataloaders,
)

from src.model import create_model


def prepare_output_directories() -> None:
    """
    Create required output directories if they do not exist.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for one complete epoch.
    """

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct_predictions += (
            (predictions == labels)
            .sum()
            .item()
        )

        total_samples += labels.size(0)

    epoch_loss = (
        running_loss / total_samples
    )

    epoch_accuracy = (
        correct_predictions / total_samples
    )

    return epoch_loss, epoch_accuracy


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation dataset.
    """

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct_predictions += (
            (predictions == labels)
            .sum()
            .item()
        )

        total_samples += labels.size(0)

    epoch_loss = (
        running_loss / total_samples
    )

    epoch_accuracy = (
        correct_predictions / total_samples
    )

    return epoch_loss, epoch_accuracy


def train_model(
    model: nn.Module,
    train_loader,
    test_loader,
    num_epochs: int = NUM_EPOCHS,
) -> Dict[str, list]:
    """
    Train and validate the FER2013 emotion recognition model.
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
        )

        val_loss, val_accuracy = validate_one_epoch(
            model,
            test_loader,
            criterion,
            DEVICE,
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        history["learning_rate"].append(current_lr)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_model_path = MODEL_DIR / "best_model.pth"

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            print(
                f"Best model saved: {best_model_path}"
            )

    return history


def main() -> None:
    """
    Execute the complete training pipeline.
    """

    prepare_output_directories()

    print(f"Using device: {DEVICE}")

    train_dataset, test_dataset = load_datasets()

    validate_class_mapping(
        train_dataset,
        test_dataset,
    )

    train_loader, test_loader = create_dataloaders(
        train_dataset,
        test_dataset,
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Testing samples: {len(test_dataset)}"
    )

    model = create_model()

    model = model.to(DEVICE)

    history = train_model(
        model,
        train_loader,
        test_loader,
    )

    print("\nTraining completed.")

    return history


if __name__ == "__main__":
    main()