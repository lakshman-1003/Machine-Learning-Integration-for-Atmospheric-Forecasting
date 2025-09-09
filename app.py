import json
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, FloatField, SubmitField
from wtforms.validators import DataRequired, Email
from src.config import Config
from src.models.db import Base, engine, SessionLocal
from src.models.user import User
from src.models.prediction import Prediction
from src.utils.auth import hash_password, verify_password
from src.services.train_ml import train_models
from src.services.train_dl import train_keras_regressor
from src.services.optimizer import train_optimizer, suggest_config
from src.services.forecasting import forecast

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config.from_object(Config)
Base.metadata.create_all(bind=engine)

def get_db():
    return SessionLocal()

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Register')

class PredictForm(FlaskForm):
    temp_c = FloatField("Temperature (°C)", validators=[DataRequired()])
    humidity = FloatField("Humidity (%)", validators=[DataRequired()])
    wind_kph = FloatField("Wind Speed (kph)", validators=[DataRequired()])
    pressure_hpa = FloatField("Pressure (hPa)", validators=[DataRequired()])
    rain_mm = FloatField("Rainfall (mm)", validators=[DataRequired()])
    submit = SubmitField("Predict")

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        with get_db() as db:
            if db.query(User).filter_by(email=form.email.data).first():
                flash('Email already registered','warning')
            else:
                u = User(email=form.email.data, name=form.name.data, password_hash=hash_password(form.password.data), is_admin=False)
                db.add(u); db.commit()
                flash('Registered successfully. Please login.','success')
                return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        with get_db() as db:
            u = db.query(User).filter_by(email=form.email.data).first()
            if u and verify_password(u.password_hash, form.password.data):
                session['user_id'] = u.id
                session['is_admin'] = bool(u.is_admin)
                session['name'] = u.name or u.email
                return redirect(url_for('dashboard'))
            flash('Invalid credentials','danger')
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', name=session.get('name'), is_admin=session.get('is_admin', False))

@app.route('/train', methods=['POST'])
def train():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    data_csv = Path(__file__).resolve().parents[1] / 'data/samples/weather_sample.csv'
    models_dir = Path(__file__).resolve().parents[1] / 'models'
    report_ml = train_models(data_csv, models_dir)
    report_dl = train_keras_regressor(data_csv, models_dir)
    flash(f'Training complete. ML: {report_ml}. DL: {report_dl}', 'success')
    return redirect(url_for('dashboard'))

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    form = PredictForm()
    pred = None
    if form.validate_on_submit():
        features = {
            'temp_c': form.temp_c.data,
            'humidity': form.humidity.data,
            'wind_kph': form.wind_kph.data,
            'pressure_hpa': form.pressure_hpa.data,
            'rain_mm': form.rain_mm.data
        }
        models_dir = Path(__file__).resolve().parents[1] / 'models'
        try:
            pred = forecast(features, models_dir)
            with get_db() as db:
                p = Prediction(features_json=json.dumps(features), output_json=json.dumps(pred))
                db.add(p); db.commit()
        except Exception as e:
            flash(f'Prediction failed: {e}', 'danger')
    return render_template('predict.html', form=form, pred=pred)

if __name__ == '__main__':
    app.run(debug=True)
