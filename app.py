# app.py
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import tempfile
import os

# Set page title
st.title(" AI Image Captioning with Voice by Raj")

# File uploader for image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=" Uploaded Image", use_column_width=True)

    # Load BLIP model
    st.text(" Loading model and generating caption by ..")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    # Generate caption
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    st.success(f" Caption: {caption}")

    # Convert caption to voice using gTTS
    tts = gTTS(text="This image looks like " + caption, lang='en')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        audio_path = tmpfile.name
        tts.save(audio_path)
        st.audio(audio_path, format='audio/mp3')
