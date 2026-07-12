# pyrefly: ignore [missing-import]
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from datetime import datetime
import pandas as pd
import io
from app.schemas.request_schema import NetworkFlowData
from app.schemas.response_schema import PredictionResponse, BatchPredictionResponse, HealthResponse
from app.services.prediction_service import process_single_prediction, process_batch_predictions, MODEL_VERSION
from live_capture.interface_detector import get_available_interfaces
from live_capture.nfstream_engine import LiveCaptureEngine

router = APIRouter()
capture_engine = LiveCaptureEngine()

@router.get("/", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint to ensure the API and Model are loaded and responsive."""
    return HealthResponse(
        status="Healthy",
        timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        model_version=MODEL_VERSION,
        message="Network Intrusion Detection System (NIDS) API is running optimally."
    )

@router.get("/metadata", tags=["System"])
def get_model_metadata():
    """Returns metadata about the active ML model."""
    return {
        "model_name": "NetworkIntrusionModel",
        "version": MODEL_VERSION,
        "description": "AI-Powered NIDS for detecting cyber attacks from network flow data.",
        "author": "Manthan Jikdara",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    }

@router.get("/interfaces", tags=["Live Monitoring"])
def get_interfaces():
    """Returns a list of available network interfaces for live monitoring."""
    return get_available_interfaces()

@router.post("/monitor/start", tags=["Live Monitoring"])
def start_monitoring(interface: str):
    """Starts live packet capture on the specified interface."""
    if capture_engine.is_running:
        raise HTTPException(status_code=400, detail="Monitoring is already running.")
        
    try:
        # Pass the process_single_prediction directly as the callback
        capture_engine.start_capture(interface=interface, callback=process_single_prediction)
        return {"status": "success", "message": f"Started monitoring interface {interface}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitor/stop", tags=["Live Monitoring"])
def stop_monitoring():
    """Stops live packet capture."""
    if not capture_engine.is_running:
        return {"status": "success", "message": "Monitoring was not running."}
        
    capture_engine.stop_capture()
    return {"status": "success", "message": "Stopped monitoring."}

@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(data: NetworkFlowData):
    """
    Predicts the threat class of a single network flow.
    Takes a structured JSON payload matching the exact training schema.
    """
    try:
        # Convert pydantic model to dict, using aliases so keys match the model's expected column names
        features = data.dict(by_alias=True)
        return process_single_prediction(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@router.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(file: UploadFile = File(...)):
    """
    Predicts threat classes for a batch of network flows via CSV upload.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Convert dataframe to list of dicts
        features_list = df.to_dict(orient="records")
        return process_batch_predictions(features_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch prediction: {str(e)}")
