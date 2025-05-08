import sounddevice as sd
from scipy.io.wavfile import write
import torch
import torch.nn as nn
import numpy as np
import librosa
import joblib
import os

# === Model definition (must match training)
class AudioFeatureMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# === Feature extraction
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(contrast, axis=1),
        np.mean(zcr, axis=1),
        np.mean(rms, axis=1),
    ])
    return features.astype(np.float32).reshape(1, -1)

# === Load song ID mapping
def load_id_mapping(mapping_file="song_id_mapping.txt"):
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            if ": " in line:
                id_str, name = line.strip().split(": ")
                mapping[int(id_str)] = name
    return mapping

# === Record 10-second clip
def record_audio(filename="mic_clip.wav", duration=10, sample_rate=22050):
    print(f"🎙️ Recording {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    write(filename, sample_rate, audio)
    print(f"✅ Saved recording to {filename}")
    return filename

# === Predict from WAV file
def predict(file_path):
    features = extract_audio_features(file_path)

    scaler = joblib.load("scaler.pkl")
    features_scaled = scaler.transform(features)

    id_to_name = load_id_mapping()
    input_dim = features_scaled.shape[1]
    num_classes = len(id_to_name)

    model = AudioFeatureMLP(input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(torch.load("audio_mlp_model.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        output = model(x_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    song_name = id_to_name.get(predicted_class, "Unknown").replace("_", " ").title()
    print(f"🎧 Predicted: {predicted_class:03d} → {song_name}")

# === Main
if __name__ == "__main__":
    wav_file = record_audio()
    predict(wav_file)
