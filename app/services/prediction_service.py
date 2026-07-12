from datetime import datetime
from app.prediction.prediction_pipeline import PredictionPipeline
from app.schemas.response_schema import PredictionResponse, BatchPredictionResponse

# Initialize the pipeline globally so it's loaded once on startup
pipeline = PredictionPipeline()

# Define the current model version
MODEL_VERSION = "1.0.0"

def calculate_risk_level(prediction: str) -> str:
    """Maps the attack class to a risk level."""
    prediction = prediction.lower()
    if "normal" in prediction or "benign" in prediction:
        return "Low"
    elif "bot" in prediction or "ddos" in prediction or "dos" in prediction or "heartbleed" in prediction or "infiltration" in prediction:
        return "Critical"
    elif "brute force" in prediction or "sql injection" in prediction or "web attack" in prediction:
        return "High"
    elif "portscan" in prediction:
        return "Medium"
    else:
        return "High" # Default for unknown attacks

def process_single_prediction(features: dict) -> PredictionResponse:
    """Processes a single prediction and enriches it with metadata."""
    result = pipeline.predict_single(features)
    prediction = result["prediction"]
    confidence = result["confidence"]
    
    return PredictionResponse(
        prediction=prediction,
        confidence=round(confidence, 2),
        risk_level=calculate_risk_level(prediction),
        model_version=MODEL_VERSION,
        timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    )

def process_batch_predictions(features_list: list) -> BatchPredictionResponse:
    """Processes a batch of predictions."""
    results = pipeline.predict_batch(features_list)
    
    prediction_responses = []
    for result in results:
        prediction_responses.append(
            PredictionResponse(
                prediction=result["prediction"],
                confidence=round(result["confidence"], 2),
                risk_level=calculate_risk_level(result["prediction"]),
                model_version=MODEL_VERSION,
                timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        )
        
    return BatchPredictionResponse(
        total_records=len(prediction_responses),
        predictions=prediction_responses,
        model_version=MODEL_VERSION,
        timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    )
