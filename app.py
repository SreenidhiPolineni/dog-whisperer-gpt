from dog_agents import get_advice
from comet_ml import Experiment
import time  # to track inference time
import streamlit as st
import weaviate
import librosa
import numpy as np
import soundfile as sf
from embed_barks import embed_audio
from weaviate.classes.query import MetadataQuery

# Connect to Weaviate
client = weaviate.connect_to_local(port=8080, headers={"X-OpenAI-Api-Key": ""})
collection = client.collections.get("DogBark")

experiment = Experiment(
    api_key="YR7Kx4Pc6DfhFQYSCmOkgT21X",
    project_name="dog_whisperer",
    workspace="sreenidhipolineni",
    log_env_details=True,
    auto_param_logging=True,
    auto_metric_logging=True
)

# Streamlit UI
st.set_page_config(page_title="Dog Whisperer GPT", page_icon="🐶", layout="centered")
st.title("🐶 Dog Whisperer GPT")
st.subheader("Upload a bark. We'll tell you what your dog is feeling!")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded audio
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(file_path, format="audio/wav")

    # Embed and search
    with st.spinner("Analyzing bark..."):
        vector = embed_audio(file_path)
        results = collection.query.near_vector(near_vector=vector.tolist(), limit=1, return_metadata=MetadataQuery(distance=True))

    # Show result
    if results.objects:
        obj = results.objects[0]
        detected_emotion = obj.properties['label']

        st.success(f"**Emotion Detected:** {obj.properties['label'].capitalize()}")

        st.caption(f"Closest match: `{obj.properties['file_name']}`\nSimilarity: `{round(1 - obj.metadata.distance, 2) * 100}%`")
        with st.spinner("Getting expert advice..."):
             advice = get_advice(detected_emotion)
             st.markdown("🧠 **Expert Advice from the Dog Whisperer Team:**")
             st.info(advice)

        # Comet Logging
        experiment.log_other("inference_label", obj.properties['label'])
        experiment.log_metric("confidence_score", round(1 - obj.metadata.distance, 4))
        experiment.log_asset(file_path, file_name=uploaded_file.name)

    else:
        st.warning("No matching bark found.")

