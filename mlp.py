import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the scaler to use during inference

# Feature Extraction
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    # Combine all extracted features into a single vector
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(contrast, axis=1),
        np.mean(zcr, axis=1),
        np.mean(rms, axis=1),
    ])
    return features.astype(np.float32)

# Load Features and Labels from Datasets
def load_data(folders=["dataset", "noisyDataset"]):
    X = []
    y = []
    for folder in folders:
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                try:
                    label = int(file.split("_")[0])
                    features = extract_audio_features(os.path.join(folder, file))
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f"Failed on {file}: {e}")
    return np.stack(X), np.array(y)

# Define MLP Model Architecture
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

# Main Training Routine
def main():
    print("Extracting features...")
    X, y = load_data()

    # Normalize features and save the scaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved to scaler.pkl")

    # Split dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Prepare data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioFeatureMLP(input_dim=X.shape[1], num_classes=60).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training model...")

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(20):
        # Training loop
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()
                predicted = torch.argmax(preds, dim=1)
                correct += (predicted == yb).sum().item()
                total += yb.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1} → Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2%}")

    # Save trained model
    torch.save(model.state_dict(), "audio_mlp_model.pt")
    print("Model saved to audio_mlp_model.pt")

    # Save training metrics
    os.makedirs("plots", exist_ok=True)
    np.savez("plots/training_metrics.npz", train_losses=train_losses, val_losses=val_losses, val_accuracies=val_accuracies)
    print("Training metrics saved to plots/training_metrics.npz")

if __name__ == "__main__":
    main()
