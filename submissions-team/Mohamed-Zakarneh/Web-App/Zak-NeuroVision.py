import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import matplotlib.pyplot as plt

# Load your trained model
@st.cache_resource
def load_model():
    return YOLO("runs/detect/yolo_tumor_model/weights/best.pt")

model = load_model()

# Streamlit UI
st.title("🧠 Brain Tumor Detector (YOLOv8)")
st.write("Upload an MRI scan and the model will detect tumors with bounding boxes.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Run YOLOv8 prediction
    with st.spinner("Running detection..."):
        # Save temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            results = model.predict(source=tmp.name, save=False, conf=0.25, verbose=False)

        img_annotated = results[0].plot()

        st.image(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB), caption="Prediction", use_column_width=True)

        # Print prediction details
        st.subheader("Prediction Details:")
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            conf = float(box.conf[0])
            st.write(f"🧠 **Detected Class:** {label} ({conf:.2f} confidence)")
