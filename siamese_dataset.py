import torch
from torch.utils.data import Dataset
import numpy as np
from features import extract_melspectrogram  # From Step 2

class AudioSiameseDataset(Dataset):
    def __init__(self, clip_pairs, target_shape=(128, 128)):
        """
        clip_pairs: list of tuples (path1, path2, label)
        target_shape: (height, width) of spectrogram to ensure uniform input
        """
        self.clip_pairs = clip_pairs
        self.target_shape = target_shape

    def __len__(self):
        return len(self.clip_pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.clip_pairs[idx]

        # Extract features
        mel1 = extract_melspectrogram(path1)
        mel2 = extract_melspectrogram(path2)

        # Resize (pad or crop) to target shape
        mel1 = self._resize_mel(mel1)
        mel2 = self._resize_mel(mel2)

        # Convert to torch tensors with channel dimension (1, H, W)
        mel1_tensor = torch.tensor(mel1).unsqueeze(0).float()
        mel2_tensor = torch.tensor(mel2).unsqueeze(0).float()
        label_tensor = torch.tensor(label).float()

        return mel1_tensor, mel2_tensor, label_tensor

    def _resize_mel(self, mel):
        """
        Pad or crop the mel spectrogram to the target shape.
        """
        h, w = mel.shape
        target_h, target_w = self.target_shape

        # Pad if smaller
        padded = np.zeros(self.target_shape)
        padded[:min(h, target_h), :min(w, target_w)] = mel[:target_h, :target_w]

        return padded
