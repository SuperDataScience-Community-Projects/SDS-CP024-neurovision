import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_train_generator(image_dir, label_dir, batch_size=32, target_size=(224, 224)):
    # Create a DataFrame to store image paths and labels
    data = []

    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):  # Ensure it's a YOLO label file
            image_filename = label_file.replace(".txt", ".jpg")  # Assuming images are .jpg
            
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, "r") as f:
                first_line = f.readline().strip()  # Read the first line
                if first_line:
                    class_id = first_line.split()[0]  # Extract the first value (0 or 1)
                    label = "negative" if class_id == "0" else "positive"  # Convert to string

                    data.append([image_filename, label])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["filename", "label"])

    # Print label distribution
    print("Training Dataset Distribution:")
    print(df["label"].value_counts())

    train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,  # Rotate images by up to 20 degrees
    width_shift_range=0.2,  # Shift images horizontally by 20% of width
    height_shift_range=0.2,  # Shift images vertically by 20% of height
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Flip images horizontally
    fill_mode="nearest"  # Fill missing pixels
)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        directory=image_dir,
        x_col="filename",
        y_col="label",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",  # Binary classification
        subset="training"
    )

    return train_generator