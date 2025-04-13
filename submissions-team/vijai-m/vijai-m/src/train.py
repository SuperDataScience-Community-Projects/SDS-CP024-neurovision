import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from preprocessing import get_train_generator
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Define dataset paths
IMAGE_DIR = "../data/brain-tumor/train/images"
LABEL_DIR = "../data/brain-tumor/train/labels"

# Get data generator
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

train_generator = get_train_generator(IMAGE_DIR, LABEL_DIR, batch_size=BATCH_SIZE, target_size=IMG_SIZE)

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

lr_callback = ReduceLROnPlateau(
    monitor='loss',  # Monitor loss instead of epoch count
    factor=0.5,      # Reduce LR by half
    patience=3,      # Wait 3 epochs before reducing
    min_lr=1e-6      # Set a lower limit
)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
EPOCHS = 100

model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_generator),
    verbose=1,
    callbacks=[lr_callback]  # Add the learning rate scheduler callback
)

# Save the model
model.save("brain_tumor_classifier.h5")

print("Model training complete. Saved as 'brain_tumor_classifier.h5'.")