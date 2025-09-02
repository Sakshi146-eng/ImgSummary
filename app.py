# app.py
import os
import io
import json
import base64
import requests
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pytesseract
import google.generativeai as genai
from dotenv import load_dotenv

# ---------------------------
# CONFIG
# ---------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
assert GEMINI_API_KEY != "", "Please set GEMINI_API_KEY (env var or .env)."

# Windows Tesseract path (update if installed elsewhere)
TESSERACT_CMD = r"C:\Users\saksh\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

OCR_LANGS = "eng+hin"
OLLAMA_MODEL = "llava:13b"

genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------
# HELPERS
# ---------------------------
def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 5:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_for_ocr(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img = deskew(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    den = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
    thr = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

def tesseract_ocr(image_np, psm=6, oem=3, lang=OCR_LANGS):
    config = f"--oem {oem} --psm {psm}"
    return pytesseract.image_to_string(image_np, lang=lang, config=config)

def run_gemini_repair(pil_img, raw_text, prompt_mode="clean"):
    model = genai.GenerativeModel("gemini-2.5-flash")

    base_system_prompt = (
        "You are an expert OCR post-processor. "
        "You are given (1) an original document image and (2) noisy OCR text from Tesseract.\n"
        "Task: reconstruct the most accurate text you can. "
        "Preserve reading order, headers, tables (as Markdown), and line breaks. "
        "Do not invent content that is not visible in the image."
    )
    extract_system_prompt = (
        "Extract key fields from the document. Respond with strict JSON only. "
        "Use keys: 'names', 'ids', 'dates', 'addresses', 'emails', 'phones', 'stamps_or_seals'. "
        "When unknown, use null or empty arrays. Do not include commentary."
    )
    user_prompt = extract_system_prompt if prompt_mode == "extract" else base_system_prompt

    resp = model.generate_content([
        {"text": user_prompt},
        pil_img,
        {"text": "-----\nOCR (Tesseract) text:\n" + str(raw_text) + "\n-----\n"}
    ])
    return resp.text.strip()

def summarize_with_ollama(text, model=OLLAMA_MODEL):
    payload = {
        "model": model,
        "prompt": f"Summarize the following document text:\n\n{text}",
    }
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
st.set_page_config(page_title="OCR + Gemini + Ollama Pipeline", layout="centered")

st.title("📄 OCR → Gemini → Ollama Summarizer")
st.write("Upload an image, extract + clean the text, and then summarize it with `llava:13b`.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Pipeline"):
        with st.spinner("Extracting text with Tesseract..."):
            pre = preprocess_for_ocr(pil_img)
            raw_text = tesseract_ocr(pre)

        st.subheader("📌 Raw OCR Output")
        st.code(raw_text[:1000] or "No text detected.")

        with st.spinner("Cleaning text with Gemini..."):
            gemini_text = run_gemini_repair(pil_img, raw_text, prompt_mode="clean")

        st.subheader("✨ Gemini Cleaned Text")
        st.code(gemini_text[:1000])

        with st.spinner("Summarizing with Ollama..."):
            summary = summarize_with_ollama(gemini_text)

        st.subheader("📌 Final Summary (Ollama)")
        st.success(summary)
