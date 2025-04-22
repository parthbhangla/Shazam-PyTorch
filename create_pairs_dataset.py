import os
import random
from glob import glob

def load_clips_by_song(clip_dir):
    """
    Loads clips and groups them by song name.
    Example return:
    {
        'Shape_of_You': ['Shape_of_You_0.wav', 'Shape_of_You_5.wav', ...],
        ...
    }
    """
    clips_by_song = {}
    for clip_path in glob(os.path.join(clip_dir, "*.wav")):
        basename = os.path.basename(clip_path)
        # Remove timestamp from filename to get song name
        song_name = "_".join(basename.split("_")[:-1])
        if song_name not in clips_by_song:
            clips_by_song[song_name] = []
        clips_by_song[song_name].append(clip_path)
    return clips_by_song

def create_clip_pairs(clips_by_song, num_pairs=10000):
    """
    Creates positive and negative clip pairs.
    Returns list of (clip1_path, clip2_path, label) where:
        - label = 1 if same song (positive pair)
        - label = 0 if different songs (negative pair)
    """
    pairs = []
    songs = list(clips_by_song.keys())

    for _ in range(num_pairs):
        # Create positive pair (same song)
        song = random.choice(songs)
        clips = clips_by_song[song]
        if len(clips) < 2:
            continue  # skip if not enough clips
        c1, c2 = random.sample(clips, 2)
        pairs.append((c1, c2, 1))

        # Create negative pair (different songs)
        song1, song2 = random.sample(songs, 2)
        c1 = random.choice(clips_by_song[song1])
        c2 = random.choice(clips_by_song[song2])
        pairs.append((c1, c2, 0))

    return pairs

# Example usage (run this part in a script or notebook to test)
if __name__ == "__main__":
    clip_dir = "Clips"  # Adjust if your path is different
    clips_by_song = load_clips_by_song(clip_dir)
    pairs = create_clip_pairs(clips_by_song, num_pairs=5000)

    # Print a few examples
    for p in pairs[:5]:
        print(p)
