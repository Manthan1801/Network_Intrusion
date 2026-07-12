# pyrefly: ignore [missing-import]
from sqlalchemy import Column, Integer, String, Float, DateTime
from db.database import Base
from datetime import datetime

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    interface = Column(String, index=True)
    source_ip = Column(String, index=True)
    destination_ip = Column(String, index=True)
    source_port = Column(Integer)
    destination_port = Column(Integer)
    protocol = Column(String)
    
    # Prediction details
    prediction = Column(String, index=True)
    confidence = Column(Float)
    severity = Column(String, index=True)
    
    # Explainability & Performance
    top_shap_feature = Column(String, nullable=True)
    latency_ms = Column(Float)
    model_version = Column(String)
