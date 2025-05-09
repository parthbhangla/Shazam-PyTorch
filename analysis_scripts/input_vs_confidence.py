import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import joblib
from record_and_predict import extract_audio_features, load_id_mapping, AudioFeatureMLP

# CONFIGURATION
file_path = "dataset/001_000.wav"  # Change to the path of the clip you want to test
durations = [1, 3, 5, 7, 10]        # Durations (in seconds) to test
sample_rate = 22050
scaler_path = "scaler.pkl"
model_path = "audio_mlp_model.pt"

# Load trained model and scaler
scaler = joblib.load(scaler_path)
id_to_name = load_id_mapping()
num_classes = len(id_to_name)

model = AudioFeatureMLP(input_dim=34, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# Load the full audio clip
y, sr = librosa.load(file_path, sr=sample_rate)

# Measure model confidence for increasing input durations
confidences = []
for sec in durations:
    clip_y = y[:sec * sample_rate]  # Trim audio to the current duration
    if len(clip_y) < 22050:  # Skip if shorter than 1 second
        continue

    # Extract audio features manually
    mfcc = librosa.feature.mfcc(y=clip_y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=clip_y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=clip_y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(clip_y)
    rms = librosa.feature.rms(y=clip_y)

    # Combine all features into a single feature vector
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(contrast, axis=1),
        np.mean(zcr, axis=1),
        np.mean(rms, axis=1),
    ]).reshape(1, -1).astype(np.float32)

    # Scale features and make prediction
    features_scaled = scaler.transform(features)
    x_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    with torch.no_grad():
        output = model(x_tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]
        confidence = np.max(probs)  # Confidence = top predicted probability
        confidences.append(confidence)

# Plot confidence vs. input duration
plt.figure(figsize=(8, 5))
plt.plot(durations[:len(confidences)], confidences, marker='o')
plt.title("Model Confidence vs. Input Duration")
plt.xlabel("Input Duration (seconds)")
plt.ylabel("Confidence (Top Class Probability)")
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/confidence_vs_duration.png")
plt.show()
