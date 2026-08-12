"""
EfficientNetV2-S Model
======================

This module defines the EfficientNetV2-S model used for facial emotion
recognition on the FER2013 dataset.

A pretrained ImageNet model is used as the backbone, and its original
1000-class classifier is replaced with a seven-class classifier for
FER2013.
"""

from torch import nn
from torchvision.models import (
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
)

from src.config import NUM_CLASSES


def create_model(
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """
    Create an EfficientNetV2-S model for FER2013 classification.

    Parameters
    ----------
    num_classes : int, optional
        Number of emotion classes. Defaults to the FER2013
        class count defined in the project configuration.

    Returns
    -------
    nn.Module
        EfficientNetV2-S model with a modified classifier.
    """

    # Load ImageNet pretrained EfficientNetV2-S weights.
    weights = EfficientNet_V2_S_Weights.DEFAULT

    # Create the pretrained EfficientNetV2-S model.
    model = efficientnet_v2_s(
        weights=weights,
    )

    # Obtain the input feature size of the original classifier.
    classifier_input_features = model.classifier[1].in_features

    # Replace the ImageNet classifier with a FER2013 classifier.
    model.classifier[1] = nn.Linear(
        classifier_input_features,
        num_classes,
    )

    return model