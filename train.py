import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from siamese_model import SiameseCNN
from siamese_dataset import AudioSiameseDataset
from create_pairs_dataset import load_clips_by_song, create_clip_pairs

# ---------- Contrastive Loss ----------
def contrastive_loss(out1, out2, label, margin=1.0):
    euclidean_distance = torch.norm(out1 - out2, dim=1)
    loss = label * torch.pow(euclidean_distance, 2) + \
           (1 - label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
    return torch.mean(loss)

# ---------- Training Script ----------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    clips_by_song = load_clips_by_song("Clips")
    pairs = create_clip_pairs(clips_by_song, num_pairs=5000)
    dataset = AudioSiameseDataset(pairs)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Init model, optimizer
    model = SiameseCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(3):
        model.train()
        total_loss = 0.0

        for x1, x2, label in loader:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)

            out1, out2 = model(x1, x2)
            loss = contrastive_loss(out1, out2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "siamese_model.pth")
    print("✅ Model saved to siamese_model.pth")

if __name__ == "__main__":
    train()
