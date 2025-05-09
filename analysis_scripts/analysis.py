import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
import joblib
from record_and_predict import extract_audio_features, load_id_mapping, AudioFeatureMLP

# Evaluate a random subset of audio clips from the dataset
def evaluate_random_clips(clip_dir="noisyDataset", n=50):
    files = [f for f in os.listdir(clip_dir) if f.endswith(".wav")]
    sample_files = random.sample(files, min(n, len(files)))

    # Load feature scaler and class label mapping
    scaler = joblib.load("scaler.pkl")
    id_to_name = load_id_mapping()
    num_classes = len(id_to_name)

    # Load trained model
    model = AudioFeatureMLP(input_dim=34, num_classes=num_classes)
    model.load_state_dict(torch.load("audio_mlp_model.pt", map_location="cpu"))
    model.eval()

    y_true, y_pred = [], []

    for file in sample_files:
        true_label = int(file.split("_")[0])  # Assumes label is prefix in filename
        try:
            features = extract_audio_features(os.path.join(clip_dir, file))
            features_scaled = scaler.transform(features)

            # Run prediction
            with torch.no_grad():
                x_tensor = torch.tensor(features_scaled, dtype=torch.float32)
                output = model(x_tensor)
                pred_label = torch.argmax(output, dim=1).item()

            y_true.append(true_label)
            y_pred.append(pred_label)
        except Exception as e:
            print(f"Failed on {file}: {e}")
    
    return y_true, y_pred

# Save confusion matrix as an image
def save_confusion_matrix(y_true, y_pred, save_path="plots/confusion_matrix_noisy.png"):
    labels_in_data = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels_in_data)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (No Labels)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

# Save per-class accuracy as a bar chart
def save_per_class_accuracy(y_true, y_pred, save_path="plots/per_class_accuracy_noisy.png"):
    correct_per_class = Counter()
    total_per_class = Counter()

    for true, pred in zip(y_true, y_pred):
        total_per_class[true] += 1
        if true == pred:
            correct_per_class[true] += 1

    labels = sorted(total_per_class.keys())
    accuracies = [correct_per_class[l] / total_per_class[l] for l in labels]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, accuracies)
    plt.xlabel("Class ID")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy (Noisy Clips)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved per-class accuracy plot to {save_path}")

# Run the evaluation and plotting pipeline
def main():
    os.makedirs("plots", exist_ok=True)

    print("Evaluating random noisy clips...")
    y_true, y_pred = evaluate_random_clips(clip_dir="noisyDataset", n=100)

    save_confusion_matrix(y_true, y_pred)
    save_per_class_accuracy(y_true, y_pred)

if __name__ == "__main__":
    main()
