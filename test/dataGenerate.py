# Install necessary packages if not installed
# !pip install transformers librosa soundfile pandas

import os
import pandas as pd
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np

# Initialize Wav2Vec2 model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()  # disable dropout

# Function to generate embedding
def generate_embedding(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    input_values = processor(y, sampling_rate=sr, return_tensors="pt").input_values
    with torch.no_grad():
        hidden_states = model(input_values).last_hidden_state
        embedding = hidden_states.mean(dim=1)  # mean pooling
    return embedding.squeeze().numpy()

# Paths
base_dir = "dataset"   # replace with your dataset folder path
folders = {"yes": "Yes", "no": "No"}  # map folder names to labels

# Prepare data list
data = []

for folder_name, label in folders.items():
    folder_path = os.path.join(base_dir, folder_name)
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            embedding = generate_embedding(file_path)
            data.append({
                "file_name": file_name,
                "embedding": embedding.tolist(),
                "label": label
            })
            print(f"Processed {file_name}")

# Save to main.csv
csv_path = os.path.join(base_dir, "main.csv")
df = pd.DataFrame(data)
df.to_csv(csv_path, index=False)
print(f"Embeddings saved to {csv_path}")