import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
import os

# Download the actual model files from HF
segmentation_model_path = hf_hub_download(
    repo_id="marinathj/neurovision-models",
    filename="segmentation_model.keras",
    repo_type="model"
)

classification_model_path = hf_hub_download(
    repo_id="marinathj/neurovision-models",
    filename="classification_model.keras",
    repo_type="model"
)

# Load the models
segmentation_model = load_model(segmentation_model_path, compile=False)
classifier_model = load_model(classification_model_path, compile=False)

# Set Streamlit layout
st.set_page_config(page_title="NeuroVision - Brain Tumor Detection", layout="wide")

# === Preprocessing function ===
def preprocess_input_image(uploaded_file):
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return image

# === Overlay mask function ===
def get_mask_overlay(image, mask):
    mask = (mask.squeeze() > 0.5).astype(np.uint8)
    colored_mask = np.zeros_like(image)
    colored_mask[..., 0] = mask * 255  # Red channel
    overlay = cv2.addWeighted(image, 0.8, colored_mask, 0.5, 0)
    return overlay

# === App Layout ===
left, right = st.columns([1, 2])

with left:
    st.markdown("""
        <div style='text-align: left; padding-bottom: 10px;'>
            <h1 style='font-size: 2.5em;'>🧠 NeuroVision</h1>
            <p style='font-size: 1.2em; margin-top: -10px;'>AI-Powered Brain Tumor Detection from MRI Scans</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📤 Upload MRI Image")
    uploaded_file = st.file_uploader("Upload a PNG, JPG or JPEG", type=["png", "jpg", "jpeg"])

    st.markdown("""
    <div style='font-size: 1em; padding-top: 10px;'>
        <ul>
            <li>Detect tumor regions using a <b>U-Net segmentation model</b></li>
            <li>Classify tumor as <b>Benign</b> or <b>Malignant</b> (if detected)</li>
        </ul>
        <hr>
        <small>This tool is for educational and research purposes only. Not for clinical diagnosis.</small>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    image = preprocess_input_image(uploaded_file)
    image_input = np.expand_dims(image, axis=0)

    predicted_mask = segmentation_model.predict(image_input)[0]
    binary_mask = (predicted_mask > 0.5).astype(np.uint8)
    overlay = get_mask_overlay((image * 255).astype(np.uint8), predicted_mask)

    with right:
        col1, col2 = st.columns(2)

        # Show original and overlay
        with col1:
            st.markdown("#### 🧾 Original MRI")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("#### 🔬 Tumor Overlay")

            # Filter tiny tumor regions before classification
            mask_uint8 = (binary_mask.squeeze() * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_area = 300  # adjust this based on how small is too small
            significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

            # Redraw mask only with valid contours
            clean_mask = np.zeros_like(mask_uint8)
            cv2.drawContours(clean_mask, significant_contours, -1, 255, thickness=cv2.FILLED)

            # If you want to visualize only valid region overlay
            refined_overlay = get_mask_overlay((image * 255).astype(np.uint8), clean_mask / 255.0)
            st.image(refined_overlay, use_container_width=True)

        # Classification based on significant region
        st.markdown("### 🧪 Classification Result")
        if len(significant_contours) > 0:
            class_prob = classifier_model.predict(image_input)[0][0]
            label = "Malignant" if class_prob > 0.5 else "Benign"
            confidence = class_prob if class_prob > 0.5 else 1 - class_prob
            st.success(f"🧬 Tumor Detected: **{label}** with **{confidence:.2%}** confidence.")
        else:
            st.info("✅ No significant tumor detected — classification not required.")

