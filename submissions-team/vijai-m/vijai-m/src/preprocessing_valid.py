import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input

def get_validation_generator(image_dir, label_dir, batch_size=32, target_size=(224, 224)):
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
    print("Validation Dataset Distribution:")
    print(df["label"].value_counts())

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=df,
        directory=image_dir,
        x_col="filename",
        y_col="label",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",  # Binary classification
        shuffle=False  # No shuffling for validation
    )

    return val_generator