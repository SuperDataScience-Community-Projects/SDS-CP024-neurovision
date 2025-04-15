import os
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.utils import class_weight
import numpy as np

from preprocessing_pretrain import get_train_generator

# Define dataset paths
IMAGE_DIR = "../data/brain-tumor/train/cropmainimages"
LABEL_DIR = "../data/brain-tumor/train/labels"

# Params
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

# Get train generator
train_generator = get_train_generator(IMAGE_DIR, LABEL_DIR, batch_size=BATCH_SIZE, target_size=IMG_SIZE)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weights_dict)

# Load base model (frozen)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

# Compile (initial training)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Initial training (with frozen base)
print("📦 Starting initial training with frozen base...")
model.fit(train_generator,
          epochs=INITIAL_EPOCHS,
          steps_per_epoch=len(train_generator),
          class_weight=class_weights_dict,
          verbose=1)

# 🔓 Unfreeze top layers of the base model
base_model.trainable = True

# 👇 Optional: Freeze lower layers and fine-tune only top layers
# Example: Unfreeze top 50 layers
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fine-tune the model
print("🔧 Starting fine-tuning with partially unfrozen base model...")
model.fit(train_generator,
          epochs=TOTAL_EPOCHS,
          initial_epoch=INITIAL_EPOCHS,  # Continue from last epoch
          steps_per_epoch=len(train_generator),
          class_weight=class_weights_dict,
          verbose=1)

# Save fine-tuned model
model.save("brain_tumor_classifier_finetuned.keras")
print("✅ Fine-tuned model saved as 'brain_tumor_classifier_finetuned.keras'")