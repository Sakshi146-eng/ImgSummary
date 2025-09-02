# app_no_tesseract.py
import os
import json
import requests
import streamlit as st
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# ---------------------------
# CONFIG
# ---------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
assert GEMINI_API_KEY != "", "Please set GEMINI_API_KEY (env var or .env)."

OLLAMA_MODEL = "llava:13b"
genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------
# HELPERS
# ---------------------------
def run_gemini_ocr_and_clean(pil_img, prompt_mode="clean"):
    model = genai.GenerativeModel("gemini-2.5-flash")

    base_prompt = (
        "You are an expert OCR system. Extract text from the given document image. "
        "Preserve reading order, headers, tables (Markdown), and formatting. "
        "Do not invent content not present in the image."
    )
    extract_prompt = (
        "Extract key fields from the document. Respond with JSON. Keys: "
        "'names', 'ids', 'dates', 'addresses', 'emails', 'phones', 'stamps_or_seals'."
    )
    user_prompt = extract_prompt if prompt_mode == "extract" else base_prompt

    resp = model.generate_content([{"text": user_prompt}, pil_img])
    return resp.text.strip()

def summarize_with_ollama(text, model=OLLAMA_MODEL):
    payload = {"model": model, "prompt": f"Summarize the following document text:\n\n{text}"}
    response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
    summary = ""
    for line in response.iter_lines():
        if line:
            try:
                obj = json.loads(line.decode("utf-8"))
                if "response" in obj:
                    summary += obj["response"]
            except json.JSONDecodeError:
                pass
    return summary

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="OCR (Gemini Only) + Ollama", layout="centered")

st.title("📄 Gemini OCR → Ollama Summarizer")
st.write("Upload an image → Gemini extracts text → Ollama summarizes it.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Pipeline"):
        with st.spinner("Extracting text with Gemini..."):
            gemini_text = run_gemini_ocr_and_clean(pil_img)

        st.subheader("✨ Gemini Extracted Text")
        st.code(gemini_text[:1000])

        with st.spinner("Summarizing with Ollama..."):
            summary = summarize_with_ollama(gemini_text)

        st.subheader("📌 Final Summary (Ollama)")
        st.success(summary)
