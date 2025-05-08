# Library imports
import os
import re
from pydub import AudioSegment

### 1. Clean song names in Songs/directory
def clean_song_names(directory="songs"):
    for filename in os.listdir(directory):
        if filename.endswith(".mp3"):
            original_path = os.path.join(directory, filename)
            new_name = re.sub(r'^\[SPOTDOWNLOADER\.COM\]\s*', '', filename)
            new_name = new_name.replace(' ', '_')
            new_name = re.sub(r'[^\w\-().]', '', new_name)
            new_path = os.path.join(directory, new_name)
            os.rename(original_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

### 2. Convert all MP3 songs to WAV
def convert_songs_to_wav(directory="Songs"):
    for filename in os.listdir(directory):
        if filename.endswith(".mp3"):
            mp3_path = os.path.join(directory, filename)
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            wav_path = os.path.join(directory, wav_filename)

            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format="wav")
            os.remove(mp3_path)
            print(f"🎵 Converted: {filename} -> {wav_filename}")

### 3. Split Songs into 10-second clips
def split_songs_to_clips(source_dir="Songs", output_dir="dataset", clip_duration_sec=10):
    os.makedirs(output_dir, exist_ok=True)
    duration_ms = clip_duration_sec * 1000
    for filename in os.listdir(source_dir):
        if filename.endswith(".wav"):
            audio = AudioSegment.from_wav(os.path.join(source_dir, filename))
            base_name = os.path.splitext(filename)[0]
            for i in range((len(audio) + duration_ms - 1) // duration_ms):
                clip = audio[i*duration_ms:(i+1)*duration_ms]
                clip_filename = f"{base_name}_clip{i}.wav"
                clip.export(os.path.join(output_dir, clip_filename), format="wav")
                print(f"🎧 Saved: {clip_filename}")

### 4. Generate numeric ID mapping for songs
def create_song_id_mapping(songs_dir="Songs", mapping_file="song_id_mapping.txt"):
    mapping = {}
    next_id = 0
    for file in sorted(os.listdir(songs_dir)):
        if file.endswith(".wav"):
            clean = re.sub(r'[^a-zA-Z0-9]', '', file.replace(".wav", "")).lower()
            if clean not in mapping:
                mapping[clean] = f"{next_id:03d}"
                next_id += 1
    with open(mapping_file, "w") as f:
        for k, v in mapping.items():
            f.write(f"{v}: {k}\n")
    print("✅ Mapping saved to", mapping_file)

### 5. Rename clips using song ID mapping
def rename_clips_by_mapping(clip_dir="dataset", mapping_file="song_id_mapping.txt"):
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            id_str, cleaned_name = line.strip().split(": ")
            mapping[cleaned_name] = id_str

    def simplify(name):
        return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

    for file in os.listdir(clip_dir):
        if file.endswith(".wav") and "_clip" in file:
            match = re.match(r"(.+)_clip(\d+)\.wav", file)
            if not match:
                continue
            song_part, clip_num = match.groups()
            cleaned_song = simplify(song_part)
            if cleaned_song in mapping:
                song_id = mapping[cleaned_song]
                new_name = f"{song_id}_{int(clip_num):03d}.wav"
                os.rename(os.path.join(clip_dir, file), os.path.join(clip_dir, new_name))
                print(f"🔁 Renamed: {file} -> {new_name}")
            else:
                print(f"⚠️ Skipped (no mapping found): {file}")

### 6. Add noise to each clip in dataset
def generate_noisy_dataset(clip_dir="dataset", noise_dir="noise", output_dir="noisyDataset"):
    os.makedirs(output_dir, exist_ok=True)
    clip_files = sorted(f for f in os.listdir(clip_dir) if f.endswith(".wav"))
    noise_files = sorted(f for f in os.listdir(noise_dir) if f.endswith(".wav"))

    for clip_file in clip_files:
        clip_audio = AudioSegment.from_wav(os.path.join(clip_dir, clip_file))
        song_id, clip_id = clip_file.replace(".wav", "").split("_")

        for noise_file in noise_files:
            noise_audio = AudioSegment.from_wav(os.path.join(noise_dir, noise_file))
            if len(noise_audio) < len(clip_audio):
                noise_audio = (noise_audio * ((len(clip_audio) // len(noise_audio)) + 1))[:len(clip_audio)]
            else:
                noise_audio = noise_audio[:len(clip_audio)]

            noise_audio = noise_audio - 15  # reduce volume
            noisy = clip_audio.overlay(noise_audio)
            noise_id = os.path.splitext(noise_file)[0]
            output_name = f"{song_id}_{clip_id}_n{noise_id}.wav"
            noisy.export(os.path.join(output_dir, output_name), format="wav")
            print(f"🎙️ Created: {output_name}")

### 7. Run the full pipeline
def build_full_pipeline():
    clean_song_names()
    convert_songs_to_wav()
    split_songs_to_clips()
    create_song_id_mapping()
    rename_clips_by_mapping()
    generate_noisy_dataset()

if __name__ == "__main__":
    build_full_pipeline()
