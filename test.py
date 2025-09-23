from ultralytics import YOLO

# Load your trained model (.pt file)
model = YOLO("runs/train/aircraft_defect/weights/best.pt")

# Export to ONNX format
success = model.export(format="onnx", imgsz=640)

print("✅ Export completed!")
print("Exported model path:", success)
