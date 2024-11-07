import whisper
import sounddevice as sd
import tempfile
import wave
import logging
import argparse

# Initialize the Whisper model
# Take time the first time you run this command, as the model is downloaded from the internet
model = whisper.load_model("medium")

# Configure logging for better output control
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def record_audio(duration, samplerate):
    """Record audio from the microphone for a given duration and samplerate.

    Parameters:
    duration (int): The duration of the recording in seconds.
    samplerate (int): The sample rate of the audio data in Hz.

    Returns:
    numpy.ndarray: The recorded audio data.
    """
    logging.info("Recording...")
    audio = sd.rec(
        int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16"
    )
    sd.wait()
    logging.info("Recording complete.")
    return audio


def transcribe_audio(audio, samplerate):
    """Convert a NumPy array containing audio data to a temporary WAV file and transcribe it using the Whisper model.

    Parameters:
    audio (numpy.ndarray): The audio data to be converted and transcribed.
    samplerate (int): The sample rate in Hz of the audio data.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".wav", mode="wb", delete=True
    ) as temp_wav_file:
        with wave.open(temp_wav_file, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(samplerate)
            wav_file.writeframes(audio.tobytes())

        result = model.transcribe(temp_wav_file.name, language="en")
        logging.info(f"Transcription: {result['text']}")


def main(duration, samplerate):
    """Record and transcribe audio using the Whisper model.

    Parameters:
    duration (int): The duration of the recording in seconds.
    samplerate (int): The sample rate of the audio data in Hz.
    """
    audio_data = record_audio(duration, samplerate)
    transcribe_audio(audio_data, samplerate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record and transcribe audio.")
    parser.add_argument(
        "--duration", type=int, default=5, help="Duration of the recording in seconds."
    )
    parser.add_argument(
        "--samplerate", type=int, default=16000, help="Sample rate of the audio data."
    )
    args = parser.parse_args()

    main(duration=args.duration, samplerate=args.samplerate)
