import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import librosa

# === Model class (must match training) ===
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

# === Feature extraction ===
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
    return features.astype(np.float32)

# === Load dataset ===
def load_data(folders=["dataset", "noisyDataset"]):
    X = []
    y = []
    for folder in folders:
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                try:
                    label = int(file.split("_")[0])  # song ID
                    features = extract_audio_features(os.path.join(folder, file))
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f"⚠️ Failed on {file}: {e}")
    return np.stack(X), np.array(y)

# === Test loop ===
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
    accuracy = 100 * correct / total
    print(f"🎯 Test Accuracy: {accuracy:.2f}%")
    return accuracy

# === Main ===
def main():
    print("📦 Loading test data...")
    X, y = load_data()

    # Load saved scaler (or fit fresh if needed)
    scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else StandardScaler().fit(X)
    X = scaler.transform(X)

    # Use consistent split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioFeatureMLP(input_dim=X.shape[1], num_classes=len(set(y)))
    model.load_state_dict(torch.load("audio_mlp_model.pt", map_location=device))
    model.to(device)

    print("🧪 Evaluating model...")
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()
