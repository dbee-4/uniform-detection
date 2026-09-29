# Uniform Compliance Detection Service

A video-based computer vision service for automated uniform compliance detection using YOLOv8, ONNX Runtime, FastAPI, Docker, and DagsHub MLflow.

---

## Features

- **Video Inference API** – Upload `.mp4`, `.avi`, or `.mov` videos through the `/predict` endpoint.
- **Frame-by-Frame Detection** – Processes video frames and generates annotated output.
- **CPU-Optimized Inference** – Uses ONNX Runtime for model inference.
- **MLOps & Model Versioning** – Tracks experiments and model versions using DagsHub and MLflow.
- **Dockerized Deployment** – Runs the application in a reproducible Docker container.
- **Swagger API Documentation** – Interactive API testing through FastAPI Swagger UI.

---

## Project Structure

```text
.
├── app.py              # FastAPI application and video inference
├── version_model.py    # MLflow/DagsHub tracking and model versioning
├── export.py           # PyTorch to ONNX model export
├── infer_video.py      # Standalone video inference script
├── Dockerfile          # Docker image configuration
├── requirements.txt    # Python dependencies
├── .dockerignore       # Docker build exclusions
└── .gitignore          # Git exclusions

'''text

## Run with Docker
docker run -d -p 8000:8000 dbee4/task1:v1.0

## Open Swagger UI:
http://localhost:8000/docs

## Local Setup
git clone https://github.com/dbee-4/uniform-detection.git
cd uniform-detection

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

## Run the API:
uvicorn app:app --reload --port 8000

The project uses DagsHub + MLflow for experiment tracking, hyperparameter logging, model artifact storage, and model versioning.

DagsHub Repository: https://dagshub.com/dbee-4/uniform-detection
MLflow Tracking Dashboard: https://dagshub.com/dbee-4/uniform-detection.mlflow

Model Registry
Registered Model: UniformDetectorModel
Version: 1
Tech Stack
Model: YOLOv8
Inference: ONNX Runtime
API: FastAPI + Uvicorn
Computer Vision: OpenCV
MLOps: MLflow + DagsHub
Containerization: Docker
Model Format: ONNX
