from datetime import datetime
import time
import logging
from app.prediction.prediction_pipeline import PredictionPipeline
from app.schemas.response_schema import PredictionResponse, BatchPredictionResponse
from core.threat_engine import calculate_risk_level
from core.explainer import ThreatExplainer
from db.database import SessionLocal
from db.models import PredictionRecord
import pandas as pd
import numpy as np

# Initialize the pipeline globally so it's loaded once on startup
pipeline = PredictionPipeline()

# Initialize Explainer
try:
    feature_names = pipeline.model.preprocessor.get_feature_names_out() if hasattr(pipeline.model.preprocessor, 'get_feature_names_out') else None
    explainer = ThreatExplainer(pipeline.model.model, feature_names)
except Exception as e:
    logging.error(f"Failed to initialize explainer: {e}")
    explainer = None

# Define the current model version
MODEL_VERSION = "1.0.0"

def _save_prediction_to_db(record_data: dict):
    """Saves a prediction to the SQLite database."""
    db = SessionLocal()
    try:
        db_record = PredictionRecord(**record_data)
        db.add(db_record)
        db.commit()
    except Exception as e:
        logging.error(f"Database insertion failed: {e}")
    finally:
        db.close()

def process_single_prediction(features: dict, meta: dict = None) -> PredictionResponse:
    """Processes a single prediction and enriches it with metadata, explainability, and saves to DB."""
    start_time = time.time()
    
    # Predict
    result = pipeline.predict_single(features)
    prediction = result["prediction"]
    confidence = result["confidence"]
    
    # Calculate Severity
    severity = calculate_risk_level(prediction, confidence)
    
    # Explainability (Only for attacks)
    top_feature = None
    if severity != "Low" and explainer:
        try:
            # We need the transformed features for SHAP. The estimator returns just the prediction.
            # We will transform it manually here for explanation.
            df = pd.DataFrame([features])
            # Align features (mimic estimator)
            df.columns = df.columns.str.strip()
            drop_columns = ['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']
            cols_to_drop = [col for col in drop_columns if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            transformed = pipeline.model.preprocessor.transform(df)
            top_feature = explainer.explain_attack(transformed)
        except Exception as e:
            logging.error(f"Failed to generate explanation: {e}")
    
    latency_ms = (time.time() - start_time) * 1000
    timestamp = datetime.utcnow()
    
    # Save to database
    db_record = {
        "timestamp": timestamp,
        "interface": meta.get("interface", "API") if meta else "API",
        "source_ip": meta.get("src_ip", "Unknown") if meta else "Unknown",
        "destination_ip": meta.get("dst_ip", "Unknown") if meta else "Unknown",
        "source_port": meta.get("src_port", 0) if meta else 0,
        "destination_port": meta.get("dst_port", 0) if meta else 0,
        "protocol": meta.get("protocol", "Unknown") if meta else "Unknown",
        "prediction": prediction,
        "confidence": confidence,
        "severity": severity,
        "top_shap_feature": top_feature,
        "latency_ms": latency_ms,
        "model_version": MODEL_VERSION
    }
    _save_prediction_to_db(db_record)
    
    return PredictionResponse(
        prediction=prediction,
        confidence=round(confidence, 2),
        risk_level=severity,
        model_version=MODEL_VERSION,
        timestamp=timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

def process_batch_predictions(features_list: list) -> BatchPredictionResponse:
    """Processes a batch of predictions."""
    results = pipeline.predict_batch(features_list)
    
    prediction_responses = []
    for i, result in enumerate(results):
        prediction = result["prediction"]
        confidence = result["confidence"]
        severity = calculate_risk_level(prediction, confidence)
        
        prediction_responses.append(
            PredictionResponse(
                prediction=prediction,
                confidence=round(confidence, 2),
                risk_level=severity,
                model_version=MODEL_VERSION,
                timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        )
        
        # Save to database so it appears on the dashboard
        db_record = {
            "timestamp": datetime.utcnow(),
            "interface": "Batch API",
            "source_ip": "Batch Upload",
            "destination_ip": "Batch Upload",
            "source_port": 0,
            "destination_port": 0,
            "protocol": "Unknown",
            "prediction": prediction,
            "confidence": confidence,
            "severity": severity,
            "top_shap_feature": "Available in Live Mode",
            "latency_ms": 0.0,
            "model_version": MODEL_VERSION
        }
        _save_prediction_to_db(db_record)
        
    return BatchPredictionResponse(
        total_records=len(prediction_responses),
        predictions=prediction_responses,
        model_version=MODEL_VERSION,
        timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    )
