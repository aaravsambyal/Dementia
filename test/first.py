import whisper
import os
import ffmpeg
import subprocess

import subprocess
os.environ["PATH"] += os.pathsep + r"C:/Users/nyc-Twice/Downloads/ffmpeg-8.0-essentials_build/ffmpeg-8.0-essentials_build/bin"
# try:
#     subprocess.run(["ffmpeg", "-version"], check=True)
#     print("FFmpeg is accessible")
# except FileNotFoundError:
#     print("FFmpeg NOT found")
# Load Whisper model (you can use 'base', 'small', 'medium', 'large')
model = whisper.load_model("medium")

# Path to your audio file
audio_path = "C:/Users/nyc-Twice/Documents/SIH/Dementia/inp_files/audio2.opus"
# print(os.path.exists(audio_path))

# Transcribe audio
result = model.transcribe(audio_path)

# Print transcription
print("Transcribed Text:")
print(result["text"])