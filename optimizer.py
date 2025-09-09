import joblib
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

FEATURES = ["mpich_version","nodes","ppn","io_block_kb","hdf5_chunk_kb","stripe_count","cpu_util","mem_gb","io_throughput_mb_s"]
TARGET = "exec_time_s"

def train_optimizer(prof_csv: Path, models_dir: Path):
    df = pd.read_csv(prof_csv)
    X = df[FEATURES]; y = df[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train); X_val_s = scaler.transform(X_val)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_s, y_train)
    preds = rf.predict(X_val_s)
    mape = float(mean_absolute_percentage_error(y_val, preds))
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, models_dir / 'optimizer_rf.joblib')
    joblib.dump(scaler, models_dir / 'optimizer_scaler.joblib')
    return {'val_mape': mape}

def suggest_config(models_dir: Path, hint: dict):
    import joblib, pandas as pd, numpy as np
    rf = joblib.load(models_dir / 'optimizer_rf.joblib')
    scaler = joblib.load(models_dir / 'optimizer_scaler.joblib')
    candidates = []
    nodes_choices = [1,2,4,8]
    ppn_choices = [4,8,16]
    for n in nodes_choices:
        for p in ppn_choices:
            candidates.append({
                'mpich_version': 381,
                'nodes': n, 'ppn': p,
                'io_block_kb': 256, 'hdf5_chunk_kb': 64, 'stripe_count': 1,
                'cpu_util': hint.get('cpu_util', 70.0),
                'mem_gb': hint.get('mem_gb', 32.0),
                'io_throughput_mb_s': hint.get('io_throughput_mb_s', 600.0)
            })
    cdf = pd.DataFrame(candidates)
    Xs = scaler.transform(cdf[["mpich_version","nodes","ppn","io_block_kb","hdf5_chunk_kb","stripe_count","cpu_util","mem_gb","io_throughput_mb_s"]])
    preds = rf.predict(Xs)
    best_idx = int(preds.argmin())
    best = cdf.iloc[best_idx].to_dict()
    best['predicted_exec_time_s'] = float(preds[best_idx])
    return best
