import mlflow
import mlflow.onnx
import onnx
import os

# Set local SQLite database for MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Uniform_Compliance_Detection")

with mlflow.start_run(run_name="yolov8m_v1_init") as run:
    # 1. Log Hyperparameters & Metadata
    mlflow.log_params({
        "model_architecture": "yolov8m",
        "input_size": "640x640",
        "framework": "ONNX Runtime / PyTorch",
        "classes": ["No-Uniform", "People", "Security", "Uniform"],
        "inference_script": "infer_video.py"
    })
    
    # 2. Log Files & Artifacts
    if os.path.exists("best.pt"):
        mlflow.log_artifact("best.pt", artifact_path="pytorch_checkpoint")
    if os.path.exists("infer_video.py"):
        mlflow.log_artifact("infer_video.py", artifact_path="scripts")
    
    # 3. Load and Register ONNX Model into MLflow Registry
    if os.path.exists("best.onnx"):
        # Load the ONNX model object from disk
        onnx_model = onnx.load("best.onnx")
        
        # Log and register in MLflow
        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            artifact_path="onnx_model",
            registered_model_name="UniformDetectorModel"
        )

    print(f"Successfully registered Version 1! Run ID: {run.info.run_id}")