import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing_valid import get_validation_generator

# Paths
val_images_path = "../data/brain-tumor/valid/cropmainimages"
val_labels_path = "../data/brain-tumor/valid/labels"
MODEL_PATH = "brain_tumor_classifier_finetuned.keras"

# Load and compile the model
model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="binary_crossentropy", 
              metrics=["accuracy"])

# Create validation generator
val_generator = get_validation_generator(val_images_path, val_labels_path)

# Get predictions (probabilities)
pred_probs = model.predict(val_generator, verbose=1)

# Convert probabilities to binary predictions (threshold = 0.5)
y_pred = (pred_probs > 0.5).astype(int).flatten()

# Get true labels
y_true = val_generator.classes

# Print metrics
print(f"\n✅ Validation Accuracy: {np.mean(y_pred == y_true) * 100:.2f}%")
print(f"✅ F1 Score: {f1_score(y_true, y_pred):.4f}\n")

# Classification report
print("🔍 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()