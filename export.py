from ultralytics import YOLO

# Load your best.pt checkpoint
model = YOLO("best.pt")

# Export to ONNX format with dynamic batching support
model.export(format="onnx", dynamic=True, simplify=True)
print("Model successfully exported to ONNX format!")