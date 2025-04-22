import os
import weaviate
from embed_barks import embed_audio

# Connect to Weaviate
client = weaviate.Client("http://localhost:8080")

# Create schema once
class_obj = {
    "class": "DogBark",
    "vectorizer": "none",
    "properties": [
        {"name": "file_name", "dataType": ["text"]},
        {"name": "label", "dataType": ["text"]}
    ]
}
if "DogBark" not in [c['class'] for c in client.schema.get()["classes"]]:
    client.schema.create_class(class_obj)

# Directory with your .wav files
sample_path = "audio_samples"

for file in os.listdir(sample_path):
    if not file.endswith(".wav"):
        continue

    label = file.split("_")[0]  # From filename like "angry_1.wav"
    path = os.path.join(sample_path, file)

    print(f"Uploading {file} as {label}...")
    try:
        vector = embed_audio(path)
        client.data_object.create(
            {
                "file_name": file,
                "label": label
            },
            "DogBark",
            vector=vector.tolist()
        )
    except Exception as e:
        print(f"❌ Failed to upload {file}: {e}")
