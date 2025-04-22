import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseCNN(nn.Module):
    def __init__(self):
        super(SiameseCNN, self).__init__()

        # Shared CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32, 64, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 64, 32, 32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (B, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)   # (B, 128, 16, 16)
        )

        # Embedding head
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 64)  # Final embedding size: 64-dim
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        return out1, out2
