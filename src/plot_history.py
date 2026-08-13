import json
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

HISTORY_PATH = Path(
    "outputs/results/training_history.json"
)

OUTPUT_DIR = Path(
    "outputs/results/plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load training history
# ============================================================

with open(
    HISTORY_PATH,
    "r",
    encoding="utf-8",
) as file:
    history = json.load(file)


epochs = range(
    1,
    len(history["train_loss"]) + 1
)


# ============================================================
# 1. Loss Curve
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["train_loss"],
    label="Training Loss",
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "loss_curve.png",
    dpi=300,
)

plt.close()


# ============================================================
# 2. Accuracy Curve
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["train_accuracy"],
    label="Training Accuracy",
)

plt.plot(
    epochs,
    history["val_accuracy"],
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "accuracy_curve.png",
    dpi=300,
)

plt.close()


# ============================================================
# 3. Learning Rate Curve
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["learning_rate"],
    label="Learning Rate",
)

plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "learning_rate_curve.png",
    dpi=300,
)

plt.close()


# ============================================================
# Summary
# ============================================================

best_epoch = (
    history["val_accuracy"].index(
        max(history["val_accuracy"])
    )
    + 1
)

best_val_accuracy = max(
    history["val_accuracy"]
)

print("=" * 60)
print("TRAINING HISTORY ANALYSIS")
print("=" * 60)

print(
    f"Total epochs completed : {len(epochs)}"
)

print(
    f"Best epoch             : {best_epoch}"
)

print(
    f"Best validation acc.   : "
    f"{best_val_accuracy:.4f} "
    f"({best_val_accuracy * 100:.2f}%)"
)

print()
print("Plots saved to:")
print(OUTPUT_DIR)

print()
print("Generated files:")

print(
    OUTPUT_DIR / "loss_curve.png"
)

print(
    OUTPUT_DIR / "accuracy_curve.png"
)

print(
    OUTPUT_DIR / "learning_rate_curve.png"
)