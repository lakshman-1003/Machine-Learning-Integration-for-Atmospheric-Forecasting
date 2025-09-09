🌦️ Machine Learning Integration for Atmospheric Forecasting

This project implements a machine learning-driven framework to improve the accuracy, efficiency, and scalability of weather forecasting. The approach integrates ML models with the Low GloSea6 numerical weather prediction (NWP) system to optimize both forecast precision and computational performance.

The project is based on the research work:
📄 "Machine Learning Integration for Improved Accuracy and Efficiency in Atmospheric Forecasting," Indian Journal of Computer Science and Technology, Vol. 4, Issue 2, May-August 2025 

INDJCST_V4I2_09_52_60_1746620733.

🚀 Features

Hybrid Forecasting Framework: Combines physics-based NWP with ML bias-correction models.
Performance Optimization: Uses profiling tools (Darshan, MPI-IO, HDF5) to tune I/O operations and runtime configurations.

ML Models Used:

Naive Bayes, Logistic Regression, SVM, Decision Trees, Random Forest, Gradient Boosting
Deep Learning Models: ANN, CNN
Real-Time Forecasting: Incorporates live satellite and meteorological data.
Visualization: Forecast results with bar charts, ratio plots, and distribution charts.

User Roles:

Service Provider – Train/test models, manage datasets, monitor performance.
Administrator – Manage users and validate access.
Remote User – Access weather predictions and personal dashboard.

🏗️ System Architecture

The framework integrates ML enhancements directly into the NWP pipeline:
Data Collection – Historical climate datasets + real-time observations.
Preprocessing & Profiling – Darshan and I/O tools for system performance tracking.
Model Training – ML algorithms for optimization + DL models for forecasting.
Optimization – Predicts optimal runtime configurations.
Deployment & Monitoring – Continuous learning and real-time adaptation.

📊 Modules

Service Provider Module – Model training, testing, visualization, forecasting.
Administrator Module – User verification and system management.
Remote User Module – User registration, login, weather predictions.

✅ Testing

The system was tested with scenarios such as:
User authentication (valid/invalid credentials).
Forecast generation (Sunny, Rainy, Cloudy).
Prediction ratio visualization.
Cross-device responsiveness.
Network error handling and recovery.

📂 Repository Structure
atmospheric-ml-forecast/
│── data/                 # Datasets (historical + real-time samples)
│── models/               # Trained ML and DL models
│── src/                  # Source code implementation
│   ├── preprocessing/    # Data cleaning & feature selection
│   ├── training/         # ML & DL training scripts
│   ├── optimization/     # Performance tuning & I/O optimization
│   └── visualization/    # Charts and prediction visualization
│── docs/                 # Documentation & research paper
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation (this file)

⚙️ Installation & Usage
# Clone the repository
git clone https://github.com/your-username/atmospheric-ml-forecast.git
cd atmospheric-ml-forecast

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows

# Install dependencies
pip install -r requirements.txt

# Run the project
python src/main.py

📈 Results

Achieved 16% margin of error in predicting model runtime execution.
Improved forecast accuracy for precipitation, wind, and temperature.
Reduced I/O bottlenecks and execution time in the Low GloSea6 model.

🔮 Future Work

Expanding dataset diversity with more hardware/software configurations.
Implementing benchmark-driven cross-inference to optimize unseen scenarios.
Generalizing framework to other HPC domains like molecular dynamics and quantum chemistry.

✍️ Authors

Saravana M K
Dhanush B
Harish Potadar
Lakshman S
Roshan Zameer

📜 Citation

If you use this project or dataset in your research, please cite:

@article{saravana2025mlforecast,
  title={Machine Learning Integration for Improved Accuracy and Efficiency in Atmospheric Forecasting},
  author={Saravana M K and Dhanush B and Harish Potadar and Lakshman S and Roshan Zameer},
  journal={Indian Journal of Computer Science and Technology},
  volume={4},
  issue={2},
  pages={52--60},
  year={2025}
}
