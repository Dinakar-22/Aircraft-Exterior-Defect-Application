from ultralytics import YOLO

# Load YOLOv8 small model (pretrained)
model = YOLO("yolov8n.pt")  # 'n' = nano, lightweight

# Train model
model.train(
    data="dataset\data.yaml",
    epochs=5,       # increase for larger dataset
    imgsz=640,       # image size
    batch=16,        # adjust depending on GPU
    lr0=0.001,       # initial learning rate (was lr)
    lrf=0.01,        # final learning rate factor
    project="runs/train",  # save path
    name="aircraft_defect",
    exist_ok=True
)

print("✅ Training complete. Model saved in runs/train/aircraft_defect/weights/best.pt")
