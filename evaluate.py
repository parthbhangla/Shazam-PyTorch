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

def identify_top_k(query_embedding, song_embeddings, k=3):
    distances = []
    for song, emb in song_embeddings.items():
        dist = np.linalg.norm(query_embedding - emb)
        distances.append((song, dist))
    distances.sort(key=lambda x: x[1])
    return distances[:k]

def evaluate_model(model, song_embeddings, num_tests=200, top_k=3):
    all_clips = glob(os.path.join(CLIPS_DIR, "*.wav"))
    random.shuffle(all_clips)
    test_clips = all_clips[:num_tests]

    top1_correct = 0
    topk_correct = 0
    wrong_predictions = []

    print(f"\n🔍 Evaluating on {len(test_clips)} clips...\n")
    for clip_path in tqdm(test_clips):
        true_song = "_".join(os.path.basename(clip_path).split("_")[:-1])
        query_embedding = get_embedding(clip_path, model)
        top_matches = identify_top_k(query_embedding, song_embeddings, k=top_k)

        predicted_top1 = top_matches[0][0]
        top_k_songs = [s for s, _ in top_matches]

        if true_song == predicted_top1:
            top1_correct += 1
        if true_song in top_k_songs:
            topk_correct += 1
        else:
            wrong_predictions.append((clip_path, true_song, top_matches))

    top1_acc = top1_correct / len(test_clips)
    topk_acc = topk_correct / len(test_clips)

    print(f"\n✅ Top-1 Accuracy: {top1_acc * 100:.2f}%")
    print(f"✅ Top-{top_k} Accuracy: {topk_acc * 100:.2f}%")
    print(f"❌ Misclassifications: {len(wrong_predictions)}")

    print("\n🧪 Sample Errors:")
    for i in range(min(5, len(wrong_predictions))):
        path, actual, guesses = wrong_predictions[i]
        print(f"\nQuery: {os.path.basename(path)}")
        print(f"Actual: {actual}")
        for rank, (guess, dist) in enumerate(guesses, 1):
            print(f"  {rank}. {guess} (Distance: {dist:.4f})")

if __name__ == "__main__":
    import random
    model = load_model()
    song_embeddings = build_reference_embeddings(model)
    evaluate_model(model, song_embeddings, num_tests=200, top_k=3)
