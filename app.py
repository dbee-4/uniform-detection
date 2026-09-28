from fastapi import FastAPI, File, UploadFile, HTTPException
import onnxruntime as ort
import cv2
import numpy as np

app = FastAPI(
    title="Uniform Compliance Detection API",
    version="1.0.0",
    description="ONNX Inference Service for People & Uniform Compliance"
)

# Load ONNX session on startup
session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Class labels index matching your model setup
CLASSES = ["No-Uniform", "People", "Security", "Uniform"]

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_version": "1.0.0"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf_threshold: float = 0.25):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    # Read image contents
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # Preprocess image for YOLO ONNX input (640x640, NCHW, Normalized)
    h, w, _ = img.shape
    resized = cv2.resize(img, (640, 640))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_tensor = rgb_img.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Run inference
    outputs = session.run([output_name], {input_name: input_tensor})[0]
    
    # Process YOLO outputs shape: (1, 8, 8400) -> (4 bbox + 4 classes)
    predictions = np.squeeze(outputs).T
    
    detections = []
    scale_x, scale_y = w / 640.0, h / 640.0

    for pred in predictions:
        scores = pred[4:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])

        if confidence >= conf_threshold:
            # Box format: [x_center, y_center, width, height]
            xc, yc, bw, bh = pred[0:4]
            x1 = float((xc - bw / 2) * scale_x)
            y1 = float((yc - bh / 2) * scale_y)
            x2 = float((xc + bw / 2) * scale_x)
            y2 = float((yc + bh / 2) * scale_y)

            detections.append({
                "class_id": class_id,
                "class_name": CLASSES[class_id] if class_id < len(CLASSES) else str(class_id),
                "confidence": round(confidence, 4),
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            })

    return {"detections": detections, "total_objects": len(detections)}