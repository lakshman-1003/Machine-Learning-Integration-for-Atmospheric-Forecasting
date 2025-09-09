import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_weather_csv(csv_path):
    try:
        df = pd.read_csv(csv_path, parse_dates=['date']).sort_values('date')
    except Exception:
        df = pd.read_csv(csv_path)
    return df

def train_val_test_split(df, target_cols, test_size=0.2, val_size=0.1, random_state=42):
    # only use targets that exist
    target_cols_present = [c for c in target_cols if c in df.columns]
    X = df.drop(columns=target_cols_present, errors='ignore')
    y = df[target_cols_present].copy() if target_cols_present else df[[]].copy()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    rel_val = val_size/(1 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - rel_val, random_state=random_state)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def build_preprocessor(X):
    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_categorical_dtype(X[c])]
    transformers = []
    if numeric_features:
        transformers.append(('num', Pipeline([('scaler', StandardScaler())]), numeric_features))
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
    pre = ColumnTransformer(transformers=transformers, remainder='drop')
    return pre
