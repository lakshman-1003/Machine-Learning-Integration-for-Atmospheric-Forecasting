from pathlib import Path
from src.services.train_ml import train_models
from src.services.train_dl import train_keras_regressor

def main():
    data_csv = Path(__file__).resolve().parents[1] / 'data/samples/weather_sample.csv'
    models_dir = Path(__file__).resolve().parents[1] / 'models'
    rep_ml = train_models(data_csv, models_dir)
    rep_dl = train_keras_regressor(data_csv, models_dir)
    print('ML report:', rep_ml)
    print('DL report:', rep_dl)

if __name__ == '__main__':
    main()
