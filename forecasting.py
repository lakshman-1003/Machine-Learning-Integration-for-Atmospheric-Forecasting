import joblib
import pandas as pd
from pathlib import Path

EXPECTED_COLS = ['temp_c', 'humidity', 'wind_kph', 'pressure_hpa', 'rain_mm']

def forecast(features: dict, models_dir: Path):
    # ensure all expected cols present (fill with 0 if missing)
    X = pd.DataFrame([[features.get(c, 0) for c in EXPECTED_COLS]], columns=EXPECTED_COLS)

    result = {}
    # load classification ensemble if exists
    try:
        clf = joblib.load(Path(models_dir) / 'classifier_rf.joblib')
        prob = clf.predict_proba(X)[:,1][0] if hasattr(clf, 'predict_proba') else None
        pred = int(clf.predict(X)[0])
        result['is_rainy_prob'] = float(prob) if prob is not None else None
        result['is_rainy'] = int(pred)
    except Exception:
        result['is_rainy_prob'] = None
        result['is_rainy'] = None

    try:
        reg = joblib.load(Path(models_dir) / 'regressor_gbr.joblib')
        temp_pred = float(reg.predict(X)[0])
        result['temp_c_pred'] = temp_pred
    except Exception:
        result['temp_c_pred'] = None

    return result
