import matplotlib.pyplot as plt
from features import extract_melspectrogram

mel = extract_melspectrogram("./Data/7_Rings.mp3")
print(mel.shape)  # e.g., (128, ~216) depending on clip duration

# Visualize the spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(mel, aspect='auto', origin='lower')
plt.title('Mel Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
