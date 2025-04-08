import os
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from datetime import datetime
# import simpleaudio as sa

# ---------------------------
# Configuration
# ---------------------------
SAMPLE_RATE = 16000  # 16kHz sampling rate
CHANNELS = 1         # Mono recording
os.makedirs("data", exist_ok=True)  # Ensure data folder exists

# ---------------------------
# Global Variables
# ---------------------------
audio_segments = []   # Stores recorded audio segments

# ---------------------------
# Recording Function
# ---------------------------
def record_audio(duration):
    """Records audio and stores it in audio_segments."""
    global audio_segments

    print("\n🔴 Recording... (Ctrl+C to stop early)")
    try:
        audio = []
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS) as stream:
            for _ in range(int(duration * SAMPLE_RATE / 512)):
                data, _ = stream.read(512)
                audio.append(data)
                # Real-time waveform display
                print("▌" * int(np.mean(np.abs(data)) * 50), end="\r")
        
        audio_segments = np.concatenate(audio)
        print("\n⏹️ Recording completed!")
    except KeyboardInterrupt:
        print("\n⏹️ Recording stopped by user.")
        audio_segments = np.concatenate(audio)

def save_recording():
    """Save the recorded audio to a file."""
    global audio_segments

    if len(audio_segments) == 0:
        print("⚠️ No recording available to save.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"data/recording_{timestamp}.wav"
    write(file_path, SAMPLE_RATE, audio_segments)
    
    print(f"\n✅ Recording saved as {file_path}")
    return file_path

# def play_audio(file_path):
#     """Play the saved audio file using simpleaudio."""
#     try:
#         if not os.path.exists(file_path):
#             print(f"⚠️ File not found: {file_path}")
#             return
        
#         print("\n🔊 Playing recorded audio...")
#         wave_obj = sa.WaveObject.from_wave_file(file_path)
#         play_obj = wave_obj.play()
#         play_obj.wait_done()  # Wait until playback finishes
#         print("🎵 Playback finished.")
#     except Exception as e:
#         print(f"⚠️ Error playing audio: {e}")

# ---------------------------
# Main Program Interface
# ---------------------------
def main():
    print("\n🎙️ Python Voice Recorder with Playback")
    print("-------------------------------------")
    
    while True:
        command = input("\nEnter command: [record/save/play/exit]: ").lower()

        if command == "record":
            duration = int(input("Enter recording time in seconds: "))
            record_audio(duration)
        elif command == "save":
            file_path = save_recording()
            if file_path:
                print(f"Saved recording at {file_path}")
        elif command == "play":
            file_path = input("Enter the path of the saved recording (or press Enter to use the latest): ")
            if not file_path.strip():
                file_path = f"data/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            play_audio(file_path)
        elif command == "exit":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid command. Please enter 'record', 'save', 'play', or 'exit'.")

if __name__ == "__main__":
    main()
