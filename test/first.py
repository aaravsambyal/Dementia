import whisper
import os
import ffmpeg

# import subprocess
from transformers import Wav2Vec2Model, Wav2Vec2Processor, BertTokenizer, BertModel
import librosa
import torch

import subprocess

os.environ["PATH"] += (
    os.pathsep
    + r"C:/Users/nyc-Twice/Downloads/ffmpeg-8.0-essentials_build/ffmpeg-8.0-essentials_build/bin"
)
# try:
#     subprocess.run(["ffmpeg", "-version"], check=True)
#     print("FFmpeg is accessible")
# except FileNotFoundError:
#     print("FFmpeg NOT found")
model = whisper.load_model("medium")

# Path to audio file
audio_path = "C:/Users/nyc-Twice/Documents/SIH/Dementia/inp_files/audio5.opus"
# print(os.path.exists(audio_path))

# Transcribe audio
result = model.transcribe(audio_path, language="en")

# Print transcription
print("Transcribed Text:")
print(result["text"])

# load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

waveform, sr = librosa.load(audio_path, sr=16000)

# preprocess
inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

# get embeddings

with torch.no_grad():
    outputs = model(**inputs)
    embeddings_audio = outputs.last_hidden_state

print(f"audio embeddings: {embeddings_audio.shape}")

# text embeddings

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertModel.from_pretrained("bert-base-multilingual-cased")

# tokenize

inputs_text = tokenizer(
    result["text"], return_tensors="pt", padding=True, truncation=True
)

# get embeddings

with torch.no_grad():
    outputs_text = model(**inputs_text)
    text_embeddings = outputs_text.last_hidden_state

sentence_embeddings = text_embeddings.mean(dim=1)
print(f"text embeddings: {text_embeddings.shape}")
# print(text_embeddings)
