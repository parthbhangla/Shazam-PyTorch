import os
import librosa
import soundfile as sf
from tqdm import tqdm

SRC_DIR = 'Data'
OUT_DIR = 'Clips'
CLIP_LEN = 5  # seconds

os.makedirs(OUT_DIR, exist_ok=True)

for filename in tqdm(os.listdir(SRC_DIR)):
    if filename.endswith('.mp3'):
        song_path = os.path.join(SRC_DIR, filename)
        song_name = os.path.splitext(filename)[0].replace(" ", "_")
        y, sr = librosa.load(song_path, sr=22050)
        total_secs = int(librosa.get_duration(y=y, sr=sr))
        
        for i in range(0, total_secs - CLIP_LEN, CLIP_LEN):
            clip = y[i * sr:(i + CLIP_LEN) * sr]
            out_path = os.path.join(OUT_DIR, f"{song_name}_{i}.wav")
            sf.write(out_path, clip, sr)
