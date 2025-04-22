import tensorflow_hub as hub
import numpy as np
import librosa

# Load the YAMNet model once
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def embed_audio(file_path):
    """
    Converts a .wav file into a 1024-dimension audio embedding using YAMNet.
    """
    waveform, sr = librosa.load(file_path, sr=16000)
    scores, embeddings, _ = yamnet_model(waveform)
    return np.mean(embeddings.numpy(), axis=0)
