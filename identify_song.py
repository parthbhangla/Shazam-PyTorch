import os
import torch
import numpy as np
from features import extract_melspectrogram
from siamese_model import SiameseCNN
from glob import glob
from tqdm import tqdm

CLIPS_DIR = "Clips"
MODEL_PATH = "siamese_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = SiameseCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def get_embedding(path, model, target_shape=(128, 128)):
    mel = extract_melspectrogram(path)
    mel = np.pad(mel, [(0, max(0, target_shape[0] - mel.shape[0])),
                       (0, max(0, target_shape[1] - mel.shape[1]))], mode='constant')
    mel = mel[:target_shape[0], :target_shape[1]]
    tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        embedding = model.forward_once(tensor)
    return embedding.squeeze().cpu().numpy()

def build_reference_embeddings(model, num_clips_per_song=5):
    clips_by_song = {}
    for clip_path in glob(os.path.join(CLIPS_DIR, "*.wav")):
        basename = os.path.basename(clip_path)
        song_name = "_".join(basename.split("_")[:-1])
        clips_by_song.setdefault(song_name, []).append(clip_path)

    song_embeddings = {}
    for song, clips in tqdm(clips_by_song.items(), desc="Building reference embeddings"):
        selected = clips[:num_clips_per_song]
        embeddings = [get_embedding(c, model) for c in selected]
        song_embeddings[song] = np.mean(embeddings, axis=0)
    return song_embeddings

def identify_song_top_k(query_clip_path, song_embeddings, model, top_k=3):
    query_embedding = get_embedding(query_clip_path, model)

    distances = []
    for song, emb in song_embeddings.items():
        dist = np.linalg.norm(query_embedding - emb)
        distances.append((song, dist))

    distances.sort(key=lambda x: x[1])
    return distances[:top_k]

if __name__ == "__main__":
    model = load_model()
    print("✅ Model loaded.")

    print("⏳ Indexing reference songs...")
    song_embeddings = build_reference_embeddings(model)

    query_clip = "Clips/Shape_of_You_15.wav"  # change as needed

    top_matches = identify_song_top_k(query_clip, song_embeddings, model, top_k=3)

    print(f"\n🎧 Query clip: {os.path.basename(query_clip)}")
    print("🎯 Top matches:")
    for i, (song, dist) in enumerate(top_matches, 1):
        print(f"  {i}. {song}  (Distance: {dist:.4f})")
