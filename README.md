Atmospheric ML Forecast - Final fixed project
Run:
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.seed_data
python -m scripts.run_training
python -m src.app
