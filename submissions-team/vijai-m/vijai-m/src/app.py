import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load the YOLO model once
model_path = "/Users/vimu/Documents/Data Science/SDS/Github/Brain-tumorCNN/runs/detect/brain_tumor_yolov8315/weights/best.pt"
model = YOLO(model_path)

def process_image(input_img):
    """Takes a PIL image from Gradio input, returns a PIL image with YOLO detections."""
    
    # Convert PIL to numpy array
    img_np = np.array(input_img.convert("L"))  # Convert to grayscale

    # Step 1: Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img_np)

    # Step 2: Apply color map (false coloring)
    color_image = cv2.applyColorMap(enhanced_img, cv2.COLORMAP_JET)

    # Step 3: Resize for model
    resized_img = cv2.resize(color_image, (256, 256))

    # Step 4: Inference with YOLO
    results = model(resized_img, conf=0.05)

    # Step 5: Get result image (with bounding boxes)
    result_img = results[0].plot()  # This returns a numpy array with annotations

    # Convert back to PIL for Gradio
    return Image.fromarray(result_img)

# Gradio Interface
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload Brain MRI"),
    outputs=gr.Image(type="pil", label="Detected Tumor (YOLO Output)"),
    title="Brain Tumor Detection (YOLO + Colorized MRI)",
    description="Upload a grayscale brain MRI image. It will be enhanced with CLAHE, colorized, and then YOLO will detect tumor regions."
)

demo.launch()