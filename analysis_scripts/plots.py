import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment
import librosa

# CONFIGURATION
dataset_path = "dataset"
noisy_dataset_path = "noisyDataset"
plots_path = "plots"
os.makedirs(plots_path, exist_ok=True)

# FEATURE EXTRACTION FUNCTION
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    # Combine all features into one vector
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(contrast, axis=1),
        np.mean(zcr, axis=1),
        np.mean(rms, axis=1),
    ])
    return features.astype(np.float32)

# === 1. PLOT CLASS DISTRIBUTION ===
def plot_class_distribution():
    clip_counts = Counter()
    for file in os.listdir(dataset_path):
        if file.endswith(".wav"):
            song_id = file.split("_")[0]  # Extract class ID
            clip_counts[song_id] += 1

    sorted_ids = sorted(clip_counts.keys())
    counts = [clip_counts[sid] for sid in sorted_ids]

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_ids, counts)
    plt.xlabel("Song ID")
    plt.ylabel("Number of Clips")
    plt.title("Class Distribution in Dataset")
    plt.tight_layout()
    plt.savefig(f"{plots_path}/class_distribution.png")
    plt.close()
    print("Saved class distribution plot.")

# PLOT FEATURE SPACE USING PCA & t-SNE
def plot_feature_space():
    X, y = [], []
    for i, file in enumerate(os.listdir(dataset_path)):
        if file.endswith(".wav") and i < 300:  # Limit for performance
            try:
                label = int(file.split("_")[0])
                feat = extract_audio_features(os.path.join(dataset_path, file))
                X.append(feat)
                y.append(label)
            except:
                continue

    X = np.stack(X)
    y = np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab20', s=10)
    plt.title("PCA of Audio Features")
    plt.colorbar(label="Class ID")
    plt.tight_layout()
    plt.savefig(f"{plots_path}/pca_visualization.png")
    plt.close()
    print("Saved PCA plot.")

    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab20', s=10)
    plt.title("t-SNE of Audio Features")
    plt.colorbar(label="Class ID")
    plt.tight_layout()
    plt.savefig(f"{plots_path}/tsne_visualization.png")
    plt.close()
    print("Saved t-SNE plot.")

# COMPARE CLEAN VS NOISY AUDIO WAVEFORMS
def plot_waveform_comparison():
    example_id = None
    for file in os.listdir(noisy_dataset_path):
        if file.endswith(".wav") and "_n" in file:
            example_id = file.split("_n")[0]
            noise_id = file.split("_n")[1].replace(".wav", "")
            break

    if not example_id:
        print("No example found for waveform comparison.")
        return

    # Build file paths for clean and noisy versions
    clean_file = f"{example_id}.wav"
    noisy_file = f"{example_id}_n{noise_id}.wav"
    clean_path = os.path.join(dataset_path, clean_file)
    noisy_path = os.path.join(noisy_dataset_path, noisy_file)

    clean_audio = AudioSegment.from_wav(clean_path)
    noisy_audio = AudioSegment.from_wav(noisy_path)

    clean_samples = np.array(clean_audio.get_array_of_samples())
    noisy_samples = np.array(noisy_audio.get_array_of_samples())

    # Trim to same length
    min_len = min(len(clean_samples), len(noisy_samples))
    clean_samples = clean_samples[:min_len]
    noisy_samples = noisy_samples[:min_len]
    residual = noisy_samples - clean_samples

    time_axis = np.linspace(0, len(clean_samples) / clean_audio.frame_rate, num=min_len)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, clean_samples)
    plt.title("Clean Audio Waveform")

    plt.subplot(3, 1, 2)
    plt.plot(time_axis, noisy_samples)
    plt.title("Noisy Audio Waveform")

    plt.subplot(3, 1, 3)
    plt.plot(time_axis, residual)
    plt.title("Residual (Noise Only)")

    plt.tight_layout()
    plt.savefig(f"{plots_path}/waveform_comparison.png")
    plt.close()
    print(f"Saved waveform comparison for clip {example_id}.")

# MAIN EXECUTION
if __name__ == "__main__":
    plot_class_distribution()
    plot_feature_space()
    plot_waveform_comparison()
