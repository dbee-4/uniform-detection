from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import cv2
import numpy as np
import onnxruntime as ort
import tempfile
import os

app = FastAPI(title="Uniform Compliance Detection - Optimized Video API")

# --- 1. Thread-limited ONNX Session Initialization ---
MODEL_PATH = "best.onnx"
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2  # Limits multi-threading CPU spike
opts.inter_op_num_threads = 2
session = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

CLASSES = ["No-Uniform", "People", "Security", "Uniform"]


def process_frame(frame):
    """Preprocess single frame, run ONNX model, draw boxes."""
    h_orig, w_orig = frame.shape[:2]
    
    # Resize to 640x640 & Convert BGR to RGB for YOLO
    img_resized = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    img_input = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))  # HWC to CHW
    img_input = np.expand_dims(img_input, axis=0)   # Batch dim (1, 3, 640, 640)

    # Run Inference
    outputs = session.run(None, {input_name: img_input})
    predictions = outputs[0][0]  # Shape (8, 8400)

    boxes, confs, class_ids = [], [], []
    for i in range(predictions.shape[1]):
        scores = predictions[4:, i]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])
        
        if confidence > 0.4:
            xc, yc, w, h = predictions[:4, i]
            
            # Map coordinates back to original video frame size
            x1 = int(max(0, (xc - w / 2) * (w_orig / 640.0)))
            y1 = int(max(0, (yc - h / 2) * (h_orig / 640.0)))
            box_w = int(w * (w_orig / 640.0))
            box_h = int(h * (h_orig / 640.0))

            boxes.append([x1, y1, box_w, box_h])
            confs.append(confidence)
            class_ids.append(class_id)

    # Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confs, score_threshold=0.4, nms_threshold=0.45)
    
    if len(indices) > 0:
        for idx in indices.flatten():
            x, y, w, h = boxes[idx]
            label = f"{CLASSES[class_ids[idx]]}: {confs[idx]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    """Accepts a video file, performs optimized frame-skipping inference, and returns processed video."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video.")

    # Save uploaded video to temp file
    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_in.write(await file.read())
    temp_in.close()

    temp_out_path = temp_in.name.replace(".mp4", "_out.mp4")

    # Read Video Stream
    cap = cv2.VideoCapture(temp_in.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- 2. Downscale Frame if Video Resolution is > 720p ---
    if width > 1280:
        height = int(height * (1280 / width))
        width = 1280

    # Video Writer Output Setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_out_path, fourcc, fps, (width, height))

    frame_count = 0
    SKIP_FRAMES = 3  # --- 3. Frame Skipping: Run ONNX inference every 3rd frame ---
    last_processed_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize large frame if needed
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        if frame_count % SKIP_FRAMES == 0:
            last_processed_frame = process_frame(frame)
            out.write(last_processed_frame)
        else:
            # Reuse previous detection overlay to keep full FPS rendering without CPU load
            out.write(last_processed_frame if last_processed_frame is not None else frame)

        frame_count += 1

    cap.release()
    out.release()

    if os.path.exists(temp_in.name):
        os.remove(temp_in.name)

    return FileResponse(temp_out_path, media_type="video/mp4", filename="processed_uniform_detection.mp4")


@app.get("/health")
def health():
    return {"status": "ok"}