# app.py
import streamlit as st
import requests
import base64
from PIL import Image
import io
import json

# Set model directly
MODEL = "llava:13b"

# Function to convert image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Function to call Ollama
def summarize_image(image, model=MODEL):
    img_base64 = image_to_base64(image)

    payload = {
        "model": model,
        "prompt": "Provide a concise summary of this image.",
        "images": [img_base64],
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)

    summary = ""
    for line in response.iter_lines():
      if line:
        data = line.decode("utf-8")
        try:
            obj = json.loads(data)
            if "response" in obj:
                summary += obj["response"]
        except json.JSONDecodeError:
            pass
    return summary

# Streamlit UI
st.set_page_config(page_title="Image Summarization App", layout="centered")

st.title("🖼️ Image Summarization with Ollama")
st.write("Upload an image and get a concise summary using a Vision-Language Model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            summary = summarize_image(image)
        st.subheader("📌 Image Summary")
        st.success(summary)
