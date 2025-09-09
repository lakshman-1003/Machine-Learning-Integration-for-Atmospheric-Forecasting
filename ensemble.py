# lightweight wrapper - not necessary for core functionality
import joblib
from pathlib import Path

def load_model_if_exists(path: Path):
    try:
        return joblib.load(path)
    except Exception:
        return None
