import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Step 1: Load and colorize the grayscale image using CLAHE + color map
def enhance_and_colorize(image_path):
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray_image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray_image)

    # Convert to false color using color map
    color_image = cv2.applyColorMap(enhanced_img, cv2.COLORMAP_JET)

    # Resize to 256x256 for model
    resized_color = cv2.resize(color_image, (256, 256))

    return resized_color

# Step 2: Define image path and model path
img_path = "/Users/vimu/Documents/Data Science/SDS/Github/Brain-tumorCNN/vijai-m/data/brain-tumor/valid/_images/val_1 (11).jpg"
model_path = "/Users/vimu/Documents/Data Science/SDS/Github/Brain-tumorCNN/runs/detect/brain_tumor_yolov8315/weights/best.pt"

# Step 3: Enhance and convert the image
colorized_img = enhance_and_colorize(img_path)

# Optional: Display colorized image using OpenCV
cv2.imshow("Colorized Image", colorized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 4: Load model and run detection
model = YOLO(model_path)
results = model(colorized_img, conf=0.05)

# Step 5: Show detections
results[0].show()
print("Detected boxes:")
print(results[0].boxes)