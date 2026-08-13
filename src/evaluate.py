"""
FER2013 Facial Emotion Recognition
==================================

This module evaluates the best trained model on the
official FER2013 test dataset.

Evaluation metrics:
    - Test loss
    - Test accuracy
    - Precision
    - Recall
    - F1-score
    - Confusion matrix
    - Per-class classification report

The official FER2013 test set is used ONLY for final evaluation.
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import json

import torch

from torch import nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ============================================================
# 2. IMPORT PROJECT MODULES
# ============================================================

from src.config import (
    DEVICE,
    MODEL_DIR,
    CLASS_NAMES,
)

from src.dataset import (
    load_datasets,
    validate_class_mapping,
    create_dataloaders,
)

from src.model import create_model


# ============================================================
# 3. EVALUATE MODEL
# ============================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader,
    device: torch.device,
):
    """
    Evaluate the trained model on the official FER2013
    test dataset.

    Returns
    -------
    dict
        Evaluation results.
    """

    # --------------------------------------------------------
    # Set model to evaluation mode.
    # --------------------------------------------------------

    model.eval()

    # --------------------------------------------------------
    # Loss function.
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Statistics.
    # --------------------------------------------------------

    running_loss = 0.0

    total_samples = 0

    all_predictions = []

    all_labels = []

    # ========================================================
    # Process test batches
    # ========================================================

    for images, labels in test_loader:

        # ----------------------------------------------------
        # Move data to device.
        # ----------------------------------------------------

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Forward pass.
        # ----------------------------------------------------

        outputs = model(images)

        # ----------------------------------------------------
        # Calculate loss.
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels,
        )

        # ----------------------------------------------------
        # Accumulate loss.
        # ----------------------------------------------------

        running_loss += (
            loss.item()
            * images.size(0)
        )

        total_samples += labels.size(0)

        # ----------------------------------------------------
        # Get predictions.
        # ----------------------------------------------------

        predictions = outputs.argmax(
            dim=1
        )

        # ----------------------------------------------------
        # Store predictions and labels.
        # ----------------------------------------------------

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    # ========================================================
    # Calculate test loss
    # ========================================================

    test_loss = (
        running_loss
        / total_samples
    )

    # ========================================================
    # Calculate accuracy
    # ========================================================

    test_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    # ========================================================
    # Precision / Recall / F1
    # ========================================================

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )
    )

    # ========================================================
    # Per-class metrics
    # ========================================================

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # ========================================================
    # Confusion matrix
    # ========================================================

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    # ========================================================
    # Create results dictionary
    # ========================================================

    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    return results


# ============================================================
# 4. MAIN
# ============================================================

def main() -> None:
    """
    Run final evaluation on the official FER2013 test set.
    """

    # ========================================================
    # Display header
    # ========================================================

    print()

    print("=" * 70)

    print(
        "FER2013 FINAL MODEL EVALUATION"
    )

    print("=" * 70)

    print()

    print(
        f"Using device: {DEVICE}"
    )

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

    print()

    # ========================================================
    # Load datasets
    # ========================================================

    print(
        "Loading FER2013 datasets..."
    )

    train_dataset, test_dataset = (
        load_datasets()
    )

    # ========================================================
    # Validate class mapping
    # ========================================================

    validate_class_mapping(
        train_dataset,
        test_dataset,
    )

    print(
        "Class mapping: OK"
    )

    print()

    print(
        f"Official test samples: "
        f"{len(test_dataset):,}"
    )

    # ========================================================
    # Create test loader
    # ========================================================

    print()

    print(
        "Creating test DataLoader..."
    )

    _, _, test_loader = create_dataloaders(
        train_dataset,
        test_dataset,
        test_dataset,
    )

    print(
        f"Test batches: "
        f"{len(test_loader):,}"
    )

    # ========================================================
    # Create model
    # ========================================================

    print()

    print(
        "Creating EfficientNetV2-S model..."
    )

    model = create_model()

    # ========================================================
    # Load best model
    # ========================================================

    best_model_path = (
        MODEL_DIR
        / "best_model.pth"
    )

    print()

    print(
        "Loading best model:"
    )

    print(
        best_model_path
    )

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=DEVICE,
        )
    )

    # ========================================================
    # Move model to device
    # ========================================================

    model = model.to(
        DEVICE
    )

    # ========================================================
    # Final evaluation
    # ========================================================

    print()

    print(
        "Evaluating on official FER2013 test set..."
    )

    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=DEVICE,
    )

    # ========================================================
    # Display results
    # ========================================================

    print()

    print("=" * 70)

    print(
        "FINAL TEST RESULTS"
    )

    print("=" * 70)

    print()

    print(
        f"Test Loss       : "
        f"{results['test_loss']:.4f}"
    )

    print(
        f"Test Accuracy   : "
        f"{results['test_accuracy']:.4f}"
        f" ({results['test_accuracy'] * 100:.2f}%)"
    )

    print(
        f"Weighted Precision : "
        f"{results['weighted_precision']:.4f}"
    )

    print(
        f"Weighted Recall    : "
        f"{results['weighted_recall']:.4f}"
    )

    print(
        f"Weighted F1-score  : "
        f"{results['weighted_f1']:.4f}"
    )

    # ========================================================
    # Classification report
    # ========================================================

    print()

    print("=" * 70)

    print(
        "CLASSIFICATION REPORT"
    )

    print("=" * 70)

    report = results[
        "classification_report"
    ]

    print()

    print(
        f"{'Class':<12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1-score':>12}"
        f"{'Support':>12}"
    )

    print("-" * 60)

    for class_name in CLASS_NAMES:

        class_result = report[
            class_name
        ]

        print(
            f"{class_name:<12}"
            f"{class_result['precision']:>12.4f}"
            f"{class_result['recall']:>12.4f}"
            f"{class_result['f1-score']:>12.4f}"
            f"{int(class_result['support']):>12}"
        )

    # ========================================================
    # Confusion matrix
    # ========================================================

    print()

    print("=" * 70)

    print(
        "CONFUSION MATRIX"
    )

    print("=" * 70)

    print()

    print(
        "Classes:"
    )

    print(
        CLASS_NAMES
    )

    print()

    for row in results[
        "confusion_matrix"
    ]:

        print(row)

    # ========================================================
    # Save results
    # ========================================================

    results_dir = (
        MODEL_DIR.parent
        / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        results_dir
        / "evaluation_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    # ========================================================
    # Final message
    # ========================================================

    print()

    print("=" * 70)

    print(
        "EVALUATION COMPLETED"
    )

    print("=" * 70)

    print()

    print(
        f"Evaluation results saved to:"
    )

    print(
        results_path
    )

    print()


# ============================================================
# 5. RUN MAIN
# ============================================================

if __name__ == "__main__":

    main()