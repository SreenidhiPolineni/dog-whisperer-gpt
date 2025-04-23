import os
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from embed_barks import embed_audio

# Connect to Weaviate (v4 style)
client = weaviate.connect_to_local(
    port=8080,
    headers={"X-OpenAI-Api-Key": ""},
)

# Create collection if not exists
if not client.collections.exists("DogBark"):
    client.collections.create(
        name="DogBark",
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="file_name", data_type=DataType.TEXT),
            Property(name="label", data_type=DataType.TEXT)
        ]
    )

dog_bark_collection = client.collections.get("DogBark")

# Go through sample audio files and embed
for fname in os.listdir("audio_samples"):
    if fname.endswith(".wav"):
        label = fname.split("_")[0]
        path = os.path.join("audio_samples", fname)
        vector = embed_audio(path)
        dog_bark_collection.data.insert(
            properties={"file_name": fname, "label": label},
            vector=vector.tolist()
        )

print("✅ All sample dog barks uploaded to Weaviate!")

