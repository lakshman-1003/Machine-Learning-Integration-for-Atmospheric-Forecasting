import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///atmo_forecast.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
