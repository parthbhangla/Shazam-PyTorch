import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import time
from scipy import signal
import librosa
import noisereduce as nr

class AudioProcessor:
    def __init__(self, sample_rate=44100, duration=5):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Sampling rate in Hz
            duration: Recording duration in seconds
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.raw_file_path = "recorded_audio.wav"
        self.processed_file_path = "isolated_music.wav"
    
    def record_audio(self):
        """Record audio from the default microphone."""
        print(f"Recording {self.duration} seconds of audio...")
        
        # Record audio
        recording = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        
        # Wait until recording is complete
        sd.wait()
        print("Recording complete.")
        
        # Save the recording, overwriting if file exists
        sf.write(self.raw_file_path, recording, self.sample_rate)
        print(f"Audio saved to {self.raw_file_path}")
        
        return recording
    
    def isolate_music(self):
        """Process the recorded audio to isolate music and reduce noise."""
        print("Processing audio to isolate music...")
        
        # Load the audio file
        data, sr = librosa.load(self.raw_file_path, sr=self.sample_rate)
        
        # Step 1: Reduce stationary noise using spectral gating
        reduced_noise = nr.reduce_noise(
            y=data, 
            sr=sr,
            prop_decrease=0.75,
            stationary=True
        )
        
        # Step 2: Apply band-pass filter to focus on typical music frequency range
        # Music generally occupies 50Hz to 10kHz range
        low_cut = 50  # Hz
        high_cut = 10000  # Hz
        nyquist = 0.5 * sr
        low = low_cut / nyquist
        high = high_cut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, reduced_noise)
        
        # Step 3: Additional noise reduction pass with different parameters
        # This helps to further isolate music components
        isolated_music = nr.reduce_noise(
            y=filtered,
            sr=sr,
            prop_decrease=0.5,
            stationary=False
        )
        
        # Save the processed audio, overwriting if file exists
        sf.write(self.processed_file_path, isolated_music, sr)
        print(f"Processed audio saved to {self.processed_file_path}")
        
        return isolated_music
    
    def process(self):
        """Run the complete audio processing workflow."""
        self.record_audio()
        self.isolate_music()
        print("Audio processing complete.")
        
        # Display file info
        raw_size = os.path.getsize(self.raw_file_path) / 1024  # KB
        processed_size = os.path.getsize(self.processed_file_path) / 1024  # KB
        print(f"Raw audio file size: {raw_size:.2f} KB")
        print(f"Processed audio file size: {processed_size:.2f} KB")


if __name__ == "__main__":
    processor = AudioProcessor()
    processor.process()