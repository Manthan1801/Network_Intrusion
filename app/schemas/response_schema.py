# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from typing import List, Optional

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    risk_level: str
    model_version: str
    timestamp: str

class BatchPredictionResponse(BaseModel):
    total_records: int
    predictions: List[PredictionResponse]
    model_version: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_version: str
    message: str
