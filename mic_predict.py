import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import os
import torch
from features import extract_melspectrogram
from siamese_model import SiameseCNN
from evaluate import build_reference_embeddings, identify_top_k

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 22050
DURATION = 5  # seconds
TEMP_FILE = "mic_clip.wav"

def record_clip():
    print("🎙️ Listening... Speak now or play music near the mic!")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    wav.write(TEMP_FILE, SAMPLE_RATE, audio)
    print(f"🎧 Saved recorded audio to {TEMP_FILE}")

def get_embedding_from_clip(path, model, target_shape=(128, 128)):
    mel = extract_melspectrogram(path)
    mel = np.pad(mel, [(0, max(0, target_shape[0] - mel.shape[0])),
                       (0, max(0, target_shape[1] - mel.shape[1]))], mode='constant')
    mel = mel[:target_shape[0], :target_shape[1]]
    tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        embedding = model.forward_once(tensor)
    return embedding.squeeze().cpu().numpy()

if __name__ == "__main__":
    model = SiameseCNN().to(DEVICE)
    model.load_state_dict(torch.load("siamese_model.pth", map_location=DEVICE))
    model.eval()

    # Build reference from clips
    print("🔎 Indexing known songs...")
    song_embeddings = build_reference_embeddings(model)

    # Record from mic
    record_clip()

    # Extract embedding and predict
    query_embedding = get_embedding_from_clip(TEMP_FILE, model)
    top_matches = identify_top_k(query_embedding, song_embeddings, k=3)

    print("\n🔍 Top matches from mic input:")
    for i, (song, dist) in enumerate(top_matches, 1):
        print(f"  {i}. {song} (Distance: {dist:.4f})")
