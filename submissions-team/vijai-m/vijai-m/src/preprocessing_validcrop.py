import os
import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

def get_validation_generator(image_dir, label_dir, batch_size=32, target_size=(224, 224)):
    data = []

    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):  # YOLO format label file
            image_filename = label_file.replace(".txt", ".jpg")  # Assuming images are .jpg
            label_path = os.path.join(label_dir, label_file)
            image_path = os.path.join(image_dir, image_filename)

            with open(label_path, "r") as f:
                lines = f.readlines()

            if len(lines) > 0:
                for line in lines:
                    values = line.strip().split()
                    class_id = values[0]  # Class label
                    x_center, y_center, width, height = map(float, values[1:])

                    # Convert to pixel coordinates
                    image = cv2.imread(image_path)
                    h, w, _ = image.shape  # Get image dimensions
                    x_center, y_center, width, height = (
                        x_center * w,
                        y_center * h,
                        width * w,
                        height * h,
                    )

                    # Calculate bounding box coordinates
                    x_min = int(x_center - width / 2)
                    y_min = int(y_center - height / 2)
                    x_max = int(x_center + width / 2)
                    y_max = int(y_center + height / 2)

                    # Crop and resize the image
                    cropped_image = image[y_min:y_max, x_min:x_max]
                    cropped_image = cv2.resize(cropped_image, target_size)

                    # Save cropped image (optional)
                    cropped_filename = f"cropped_{image_filename}"
                    cv2.imwrite(os.path.join(image_dir, cropped_filename), cropped_image)

                    # Append data
                    label = "negative" if class_id == "0" else "positive"
                    data.append([cropped_filename, label])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["filename", "label"])

    # Print label distribution
    print("Validation Dataset Distribution:")
    print(df["label"].value_counts())

    # Image preprocessing
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=df,
        directory=image_dir,
        x_col="filename",
        y_col="label",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False  # No shuffling for validation
    )

    return val_generator