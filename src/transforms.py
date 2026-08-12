"""
Image Transformations
=====================

This module defines the image preprocessing pipelines used during
training and evaluation.

Different transformations are applied to the training and testing
datasets.

Training:
    - Resize
    - Data augmentation
    - Tensor conversion
    - Normalization

Testing:
    - Resize
    - Tensor conversion
    - Normalization
"""

from torchvision import transforms

from src.config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD
)


def get_train_transforms():
    """
    Create image transformations used during model training.

    Returns
    -------
    torchvision.transforms.Compose
        Image preprocessing pipeline for the training dataset.
    """

    train_transform = transforms.Compose([

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=10
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )

    ])

    return train_transform




def get_test_transforms():
    """
    Create image preprocessing used during model evaluation.

    Returns
    -------
    torchvision.transforms.Compose
        Image preprocessing pipeline for the testing dataset.
    """

    test_transform = transforms.Compose([

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )

    ])

    return test_transform