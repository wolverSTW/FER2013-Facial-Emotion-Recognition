"""
FER2013 Dataset Module
======================

This module provides functions for:

1. Loading the FER2013 dataset using torchvision ImageFolder.
2. Splitting the training dataset into training and validation subsets.
3. Keeping the official FER2013 test dataset separate for final evaluation.
4. Creating PyTorch DataLoaders.

Dataset structure:

    data/
        FER2013/
            train/
                angry/
                disgust/
                fear/
                happy/
                neutral/
                sad/
                surprise/

            test/
                angry/
                disgust/
                fear/
                happy/
                neutral/
                sad/
                surprise/

Data split:

    Original train/
        ├── 90% Training
        └── 10% Validation

    Original test/
        └── 100% Final Testing
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from typing import Tuple

import torch

from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
)

from torchvision import datasets


# ============================================================
# 2. IMPORT PROJECT CONFIGURATION
# ============================================================

from src.config import (
    TRAIN_DIR,
    TEST_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    CLASS_NAMES,
)


# ============================================================
# 3. IMPORT IMAGE TRANSFORMS
# ============================================================

from src.transforms import (
    get_train_transforms,
    get_test_transforms,
)


# ============================================================
# 4. VALIDATION SPLIT
# ============================================================

VALIDATION_SPLIT = 0.10


# ============================================================
# 5. RANDOM SEED
# ============================================================

RANDOM_SEED = 42


# ============================================================
# 6. LOAD DATASETS
# ============================================================

def load_datasets() -> Tuple[
    datasets.ImageFolder,
    datasets.ImageFolder,
]:
    """
    Load the original FER2013 training and testing datasets.

    The training dataset is loaded with training transforms.

    The test dataset is loaded with test transforms.

    Returns
    -------
    Tuple[datasets.ImageFolder, datasets.ImageFolder]
        Original training dataset and original test dataset.
    """

    # --------------------------------------------------------
    # Load training dataset.
    # --------------------------------------------------------

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=get_train_transforms(),
    )

    # --------------------------------------------------------
    # Load test dataset.
    # --------------------------------------------------------

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=get_test_transforms(),
    )

    return (
        train_dataset,
        test_dataset,
    )


# ============================================================
# 7. VALIDATE CLASS MAPPING
# ============================================================

def validate_class_mapping(
    train_dataset: datasets.ImageFolder,
    test_dataset: datasets.ImageFolder,
) -> None:
    """
    Verify that the training and testing datasets contain
    the expected FER2013 emotion classes.

    Parameters
    ----------
    train_dataset : datasets.ImageFolder
        FER2013 training dataset.

    test_dataset : datasets.ImageFolder
        FER2013 testing dataset.

    Raises
    ------
    ValueError
        If class mappings do not match.
    """

    # --------------------------------------------------------
    # Expected FER2013 classes.
    # --------------------------------------------------------

    expected_classes = CLASS_NAMES

    # --------------------------------------------------------
    # Check training classes.
    # --------------------------------------------------------

    if train_dataset.classes != expected_classes:

        raise ValueError(
            "Training dataset classes do not match "
            f"the expected FER2013 classes: "
            f"{expected_classes}"
        )

    # --------------------------------------------------------
    # Check testing classes.
    # --------------------------------------------------------

    if test_dataset.classes != expected_classes:

        raise ValueError(
            "Testing dataset classes do not match "
            f"the expected FER2013 classes: "
            f"{expected_classes}"
        )

    # --------------------------------------------------------
    # Check class-to-index mapping.
    # --------------------------------------------------------

    if (
        train_dataset.class_to_idx
        != test_dataset.class_to_idx
    ):

        raise ValueError(
            "Training and testing class mappings "
            "are different."
        )


# ============================================================
# 8. CREATE TRAIN / VALIDATION SPLIT
# ============================================================

def create_train_validation_split(
    train_dataset: datasets.ImageFolder,
) -> Tuple[Subset, Subset]:
    """
    Split the original training dataset into:

        90% Training
        10% Validation

    A fixed random seed is used to make the split reproducible.

    Parameters
    ----------
    train_dataset : datasets.ImageFolder
        Original FER2013 training dataset.

    Returns
    -------
    Tuple[Subset, Subset]
        Training subset and validation subset.
    """

    # --------------------------------------------------------
    # Calculate dataset sizes.
    # --------------------------------------------------------

    total_size = len(
        train_dataset
    )

    validation_size = int(
        total_size
        * VALIDATION_SPLIT
    )

    training_size = (
        total_size
        - validation_size
    )

    # --------------------------------------------------------
    # Create a reproducible random generator.
    # --------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Randomly split the dataset.
    # --------------------------------------------------------

    train_subset, validation_subset = (
        torch.utils.data.random_split(
            train_dataset,
            [
                training_size,
                validation_size,
            ],
            generator=generator,
        )
    )

    return (
        train_subset,
        validation_subset,
    )


# ============================================================
# 9. CREATE VALIDATION DATASET WITH TEST TRANSFORMS
# ============================================================

def create_validation_dataset(
    train_dataset: datasets.ImageFolder,
    validation_subset: Subset,
) -> Subset:
    """
    Create a validation subset using test/evaluation transforms.

    The validation data should NOT use random augmentation.

    Therefore, the validation subset uses:

        - Grayscale
        - Resize
        - ToTensor
        - Normalize

    instead of random training augmentation.

    Parameters
    ----------
    train_dataset : datasets.ImageFolder
        Original training dataset.

    validation_subset : Subset
        Validation subset produced by random_split.

    Returns
    -------
    Subset
        Validation subset using test transforms.
    """

    # --------------------------------------------------------
    # Create a separate ImageFolder dataset using test
    # transforms.
    # --------------------------------------------------------

    validation_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=get_test_transforms(),
    )

    # --------------------------------------------------------
    # Reuse exactly the same validation indices.
    # --------------------------------------------------------

    validation_subset = Subset(
        validation_dataset,
        validation_subset.indices,
    )

    return validation_subset


# ============================================================
# 10. CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """
    Create PyTorch DataLoaders for:

        1. Training
        2. Validation
        3. Testing

    Parameters
    ----------
    train_dataset : Dataset
        Training dataset.

    validation_dataset : Dataset
        Validation dataset.

    test_dataset : Dataset
        Official FER2013 test dataset.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        Training, validation and test DataLoaders.
    """

    # --------------------------------------------------------
    # Training DataLoader.
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Validation DataLoader.
    # --------------------------------------------------------

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Test DataLoader.
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )