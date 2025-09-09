import json, joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from src.services.data_prep import load_weather_csv, train_val_test_split, build_preprocessor

def train_models(data_csv: Path, models_dir: Path):
    models_dir.mkdir(parents=True, exist_ok=True)
    df = load_weather_csv(data_csv)

    target_class = 'is_rainy'
    target_reg = 'temp_c'

    # split (drop date if present)
    (X_train, y_train_df), (X_val, y_val_df), _ = train_val_test_split(df.drop(columns=['date'], errors='ignore'), target_cols=[target_class, target_reg])

    # prepare targets safely
    y_train_cls = y_train_df[target_class] if target_class in y_train_df.columns else None
    y_val_cls = y_val_df[target_class] if target_class in y_val_df.columns else None
    y_train_reg = y_train_df[target_reg] if target_reg in y_train_df.columns else None
    y_val_reg = y_val_df[target_reg] if target_reg in y_val_df.columns else None

    # features: all columns except targets + date
    feature_cols = [c for c in df.columns if c not in [target_class, target_reg, 'date']]
    pre = build_preprocessor(df[feature_cols])

    cls_reports = {}
    # only train classifiers if classification target exists
    if y_train_cls is not None:
        classifiers = [
            ('logreg', LogisticRegression(max_iter=1000)),
            ('svm', SVC(probability=True)),
            ('dt', DecisionTreeClassifier(max_depth=6)),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
        ]
        for name, mdl in classifiers:
            pipe = Pipeline([('pre', pre), ('model', mdl)])
            pipe.fit(X_train, y_train_cls)
            preds = pipe.predict(X_val)
            cls_reports[name] = {'val_accuracy': float(accuracy_score(y_val_cls, preds))}
            joblib.dump(pipe, models_dir / f'classifier_{name}.joblib')

    reg_reports = {}
    if y_train_reg is not None:
        regressors = [
            ('linr', LinearRegression()),
            ('gbr', GradientBoostingRegressor(random_state=42))
        ]
        for name, mdl in regressors:
            pipe = Pipeline([('pre', pre), ('model', mdl)])
            pipe.fit(X_train, y_train_reg)
            preds = pipe.predict(X_val)
            reg_reports[name] = {'val_mae': float(mean_absolute_error(y_val_reg, preds))}
            joblib.dump(pipe, models_dir / f'regressor_{name}.joblib')

    report = {'classification': cls_reports, 'regression': reg_reports}
    (models_dir / 'ml_report.json').write_text(json.dumps(report, indent=2))
    return report
