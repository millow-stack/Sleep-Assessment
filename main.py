import os
import logging
from datetime import datetime
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import noisereduce as nr
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# ---------------------------
# Configuration
# ---------------------------
SAMPLE_RATE = 16000  # 16kHz sampling
CHANNELS = 1         # Mono recording
NOISE_DURATION = 1   # Seconds for noise profiling

# Directories
RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
LOG_DIR = "logs"
SPEC_DIR = "data/spectrograms"

# Create necessary directories
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SPEC_DIR, exist_ok=True)

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"recorder_{datetime.now().strftime('%Y%m%d')}.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------
# Core Functions
# ---------------------------
def record_audio(duration):
    """Record audio with real-time monitoring"""
    try:
        logging.info(f"Starting {duration}s recording")
        print("\n🔴 Recording... (Ctrl+C to stop early)")
        
        audio = []
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS) as stream:
            for _ in range(int(duration * SAMPLE_RATE / 512)):
                data, _ = stream.read(512)
                audio.append(data)
                # Real-time waveform display
                print("▌" * int(np.mean(np.abs(data)) * 50), end="\r")
        
        return np.concatenate(audio)
    except KeyboardInterrupt:
        print("\n⏹️ Recording stopped by user")
        return np.concatenate(audio)
    except Exception as e:
        logging.error(f"Recording failed: {str(e)}")
        raise

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    return audio / max_val if max_val > 0 else audio

def reduce_noise(audio):
    """Apply noise reduction to recorded audio"""
    try:
        noise_sample = audio[:int(NOISE_DURATION * SAMPLE_RATE)]
        return nr.reduce_noise(
            y=audio,
            y_noise=noise_sample,
            sr=SAMPLE_RATE,
            prop_decrease=0.9
        )
    except Exception as e:
        logging.error(f"Noise reduction failed: {str(e)}")
        raise

def plot_spectrogram(audio, sample_rate, title="Spectrogram", save_path=None):
    """Plot and optionally save the spectrogram"""
    try:
        # Flatten audio if needed (handles shape like (N, 1, 1), (N, 1), etc.)
        audio = np.squeeze(audio)

        f, t, Sxx = spectrogram(audio, fs=sample_rate)
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(title)
        plt.colorbar(label='Power [dB]')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logging.info(f"Saved spectrogram: {save_path}")
        else:
            plt.show()

        plt.close()
    except Exception as e:
        logging.error(f"Spectrogram plotting failed: {str(e)}")
        raise


def save_recording(audio, clean_audio):
    """Save both raw and cleaned audio"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    raw_path = os.path.join(RAW_DIR, f"raw_{timestamp}.wav")
    clean_path = os.path.join(CLEAN_DIR, f"clean_{timestamp}.wav")
    
    write(raw_path, SAMPLE_RATE, audio.astype(np.float32))
    write(clean_path, SAMPLE_RATE, clean_audio.astype(np.float32))

    # Save spectrograms
    plot_spectrogram(audio, SAMPLE_RATE, title="Raw Audio Spectrogram",
                     save_path=os.path.join(SPEC_DIR, f"spectrogram_raw_{timestamp}.png"))
    plot_spectrogram(clean_audio, SAMPLE_RATE, title="Cleaned Audio Spectrogram",
                     save_path=os.path.join(SPEC_DIR, f"spectrogram_clean_{timestamp}.png"))
    
    return raw_path, clean_path

def play_audio(audio):
    """Play processed audio"""
    try:
        print("\n🔊 Playing cleaned audio...")
        sd.play(audio, SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        logging.error(f"Playback failed: {str(e)}")
        raise

# ---------------------------
# Main Program
# ---------------------------
def main():
    print("🎙️ Python Sleep Recorder")
    print("-----------------------")
    
    try:
        # Get recording duration
        duration = int(input("Enter recording time in minutes: ")) * 60
        if duration < 60:
            print("⚠️ Minimum 1 minute required for sleep analysis")
            duration = 60
        
        # Record audio
        raw_audio = record_audio(duration)
        raw_audio = normalize_audio(raw_audio)
        
        # Noise reduction
        clean_audio = (raw_audio)
        
        # Save files and spectrograms
        raw_path, clean_path = save_recording(raw_audio, clean_audio)
        print(f"\n✅ Saved recordings:\n- Raw: {raw_path}\n- Clean: {clean_path}")
        
        # Playback
        play_audio(clean_audio)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}\nCheck logs for details")

if __name__ == "__main__":
    main()
