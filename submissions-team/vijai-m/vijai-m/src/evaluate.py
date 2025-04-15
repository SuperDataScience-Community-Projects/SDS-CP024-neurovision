import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Paths to validation images and labels
val_images_path = "../data/brain-tumor/valid/images"
val_labels_path = "../data/brain-tumor/valid/labels"

# Model path (update if saved differently)
MODEL_PATH = "brain_tumor_classifier.h5"  # Change if your model has a different filename

# Image preprocessing settings
IMG_SIZE = (224, 224)  # Ensure this matches the input size used during training
BATCH_SIZE = 32  # Adjust based on memory

# Load trained model
model = load_model(MODEL_PATH)

# Function to extract first byte (class label) from YOLO format labels
def get_label_from_yolo(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()
        if lines:
            first_byte = int(lines[0].split()[0])  # Extract first byte (0 or 1)
            return first_byte
    return None  # Handle empty label files

# Load validation images and labels
image_files = sorted(os.listdir(val_images_path))
label_files = sorted(os.listdir(val_labels_path))

true_labels = []
image_tensors = []

for img_file, lbl_file in zip(image_files, label_files):
    img_path = os.path.join(val_images_path, img_file)
    lbl_path = os.path.join(val_labels_path, lbl_file)

    # Load and preprocess image
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img).astype("float32") / 255.0  # Normalize
    image_tensors.append(img_array)

    # Load label
    label = get_label_from_yolo(lbl_path)
    if label is not None:
        true_labels.append(label)

# Convert to NumPy arrays
image_tensors = np.array(image_tensors)
true_labels = np.array(true_labels)

# Make predictions
predictions = model.predict(image_tensors)
pred_labels = (predictions > 0.5).astype(int).flatten()  # Convert probabilities to class labels (binary)

# Calculate accuracy
accuracy = np.mean(pred_labels == true_labels)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")