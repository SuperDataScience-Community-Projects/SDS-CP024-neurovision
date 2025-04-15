from ultralytics import YOLO

# Load a YOLOv8 model (Nano version to start)
#model = YOLO('yolov8n.yaml')  # or yolov8s.yaml, yolov8m.yaml, etc.

# Load the best weights from the previous run
model = YOLO('/Users/vimu/Documents/Data Science/SDS/Github/Brain-tumorCNN/runs/detect/brain_tumor_yolov8314/weights/best.pt')



#model.train(
#    data='/Users/vimu/Documents/Data Science/SDS/Github/Brain-tumorCNN/vijai-m/data/brain-tumor/brain-tumor.yaml',
#    epochs=10,
#    imgsz=640,
#    batch=8,
#    patience=10,
#    device='cpu',
#    name='brain_tumor_yolov83'
#)


# Train the model with MRI-safe augmentations
model.train(
    data='/Users/vimu/Documents/Data Science/SDS/Github/Brain-tumorCNN/vijai-m/data/brain-tumor/brain-tumor.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    patience=10,
    device='cpu',
    name='brain_tumor_yolov83',

    # MRI-safe augmentations
    degrees=5.0,        # Small rotations to simulate head tilt
    translate=0.1,      # Slight translations
    scale=0.5,          # Minor zoom in/out
    shear=2.0,          # Subtle shearing

    # Safer flips
    fliplr=0.2,         # Light horizontal flip (common in MRI)
    flipud=0.0,         # Avoid vertical flips for brain scans

    # Disable irrelevant augmentations for grayscale MRI
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    perspective=0.0,
    mosaic=0.0,
    mixup=0.0
)