"""
FER2013 Dataset Utilities
=========================

This module provides functions for loading the FER2013 dataset and
creating PyTorch DataLoaders.

The training and testing datasets are loaded using torchvision's
ImageFolder class. Image transformations are imported from the
transforms module.
"""

from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets

from src.config import (
    TRAIN_DIR,
    TEST_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    CLASS_NAMES,
)

from src.transforms import (
    get_train_transforms,
    get_test_transforms,
)


def load_datasets() -> Tuple[
    datasets.ImageFolder,
    datasets.ImageFolder
]:
    """
    Load the FER2013 training and testing datasets.

    Returns
    -------
    Tuple[datasets.ImageFolder, datasets.ImageFolder]
        A tuple containing the training dataset and testing dataset.
    """

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=get_train_transforms(),
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=get_test_transforms(),
    )

    return train_dataset, test_dataset


def validate_class_mapping(
    train_dataset: datasets.ImageFolder,
    test_dataset: datasets.ImageFolder,
) -> None:
    """
    Verify that the training and testing datasets contain the
    expected FER2013 emotion classes.

    Parameters
    ----------
    train_dataset : datasets.ImageFolder
        FER2013 training dataset.

    test_dataset : datasets.ImageFolder
        FER2013 testing dataset.

    Raises
    ------
    ValueError
        If the class mappings differ from the expected classes.
    """

    expected_classes = CLASS_NAMES

    if train_dataset.classes != expected_classes:
        raise ValueError(
            "Training dataset classes do not match the expected "
            f"FER2013 classes: {expected_classes}"
        )

    if test_dataset.classes != expected_classes:
        raise ValueError(
            "Testing dataset classes do not match the expected "
            f"FER2013 classes: {expected_classes}"
        )

    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError(
            "Training and testing class mappings are different."
        )


def create_dataloaders(
    train_dataset: datasets.ImageFolder,
    test_dataset: datasets.ImageFolder,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and testing.

    Parameters
    ----------
    train_dataset : datasets.ImageFolder
        FER2013 training dataset.

    test_dataset : datasets.ImageFolder
        FER2013 testing dataset.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Training DataLoader and testing DataLoader.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, test_loader