from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from src.models.db import Base

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    features_json = Column(String)
    output_json = Column(String)
    model_name = Column(String, default="ensemble_v1")
