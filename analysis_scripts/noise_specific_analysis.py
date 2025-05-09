import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from record_and_predict import extract_audio_features, load_id_mapping, AudioFeatureMLP
import joblib

# Evaluate a list of audio files and return true/predicted labels
def evaluate_files(file_list, base_dir="noisyDataset"):
    scaler = joblib.load("scaler.pkl")
    id_to_name = load_id_mapping()
    num_classes = len(id_to_name)

    # Load trained model
    model = AudioFeatureMLP(input_dim=34, num_classes=num_classes)
    model.load_state_dict(torch.load("audio_mlp_model.pt", map_location="cpu"))
    model.eval()

    y_true, y_pred = [], []
    for file in file_list:
        true_label = int(file.split("_")[0])  # Assumes label is prefix in filename
        try:
            # Extract and scale features
            features = extract_audio_features(os.path.join(base_dir, file))
            features_scaled = scaler.transform(features)

            # Predict with model
            with torch.no_grad():
                x_tensor = torch.tensor(features_scaled, dtype=torch.float32)
                output = model(x_tensor)
                pred_label = torch.argmax(output, dim=1).item()

            y_true.append(true_label)
            y_pred.append(pred_label)
        except:
            continue  # Skip files that cause errors
    return y_true, y_pred

# Save confusion matrix plot for a given noise ID
def save_confusion_matrix(y_true, y_pred, noise_id):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Noise: {noise_id}")
    plt.colorbar()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/conf_matrix_{noise_id}.png")
    plt.close()

# Main loop: group files by noise type and generate a confusion matrix for each
def main():
    files_by_noise = defaultdict(list)

    # Group files based on the noise ID at the end of the filename
    for f in os.listdir("noisyDataset"):
        if f.endswith(".wav") and "_n" in f:
            noise_id = f.split("_")[-1].replace(".wav", "")
            files_by_noise[noise_id].append(f)

    # Evaluate and plot results per noise condition
    for noise_id, files in files_by_noise.items():
        y_true, y_pred = evaluate_files(files)
        if y_true and y_pred:
            save_confusion_matrix(y_true, y_pred, noise_id)

if __name__ == "__main__":
    main()
