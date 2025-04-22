import librosa
import numpy as np

def extract_melspectrogram(path, sr=22050, n_mels=128):
    """
    Load a WAV file and convert it to a Mel Spectrogram.
    Returns: 2D numpy array (n_mels x time)
    """
    y, sr = librosa.load(path, sr=sr)
    
    # Generate Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    
    # Convert power spectrogram to decibel (log scale)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    return S_dB
