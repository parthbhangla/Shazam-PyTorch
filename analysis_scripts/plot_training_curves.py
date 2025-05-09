import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# Load saved metrics
data = np.load("plots/training_metrics.npz")
train_losses = data["train_losses"]
val_losses = data["val_losses"]
val_accuracies = data["val_accuracies"]

epochs = range(1, len(train_losses) + 1)

# 1. Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/loss_vs_epoch.png")
plt.close()

# 2. Accuracy Curve
plt.figure(figsize=(8, 5))
plt.plot(epochs, val_accuracies, label="Val Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy vs Epoch")
plt.ylim(0, 1.0)
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/accuracy_vs_epoch.png")
plt.close()

print("Saved loss and accuracy curves to /plots")
