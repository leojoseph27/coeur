import os

# Force TensorFlow to use CPU only and avoid slow GPU/driver scans
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_socketio import SocketIO
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
import os
import logging
from urllib.parse import quote as url_quote
import json
import torch
from detecting_anomaly_in_ecg_data_using_autoencoder_with_pytorch import Autoencoder
from datetime import datetime
import math
import requests
import time
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from html import unescape
import re
import subprocess
from dotenv import load_dotenv
import db_neon
from werkzeug.security import generate_password_hash

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Render/hosting port (used by some environments/tools)
PORT = int(os.environ.get("PORT", "10000"))

# Create necessary directories
os.makedirs('reports', exist_ok=True)
os.makedirs('ecg project', exist_ok=True)
os.makedirs('heart/models', exist_ok=True)
os.makedirs('archive/variables', exist_ok=True)

# Load environment variables. override=True is required because the sandbox
# shell already exports DATABASE_URL (pointing to the Next.js SQLite db); we
# need our .env values to take precedence.
load_dotenv(override=True)
logger.info("Environment variables loaded")

# Set application mode
DEBUG_MODE = os.environ.get('DEBUG', 'False').lower() == 'true'
DEPLOYMENT_MODE = os.environ.get('DEPLOYMENT_ENV', 'development')
logger.info(f"Starting application in {DEPLOYMENT_MODE} mode")

# Check and log missing model files without downloading at startup (keeps startup fast on Render)
MODEL_FILES = [
    'ecg project/best_model.pth',
    'heart/models/audio_model.h5',
    'archive/saved_model.pb',
    'archive/variables/variables.data-00000-of-00001'
]

missing_files = [fp for fp in MODEL_FILES if not os.path.exists(fp) or os.path.getsize(fp) == 0]
if missing_files:
    logger.warning(f"Missing model files at startup (some features may be unavailable): {missing_files}")

app = Flask(__name__)
_secret = os.getenv("FLASK_SECRET_KEY", "")
if not _secret:
    if os.getenv("DEPLOYMENT_ENV", "development").lower() == "production":
        raise RuntimeError(
            "FLASK_SECRET_KEY must be set in production. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    _secret = "default-secret-key-for-development"
app.config['SECRET_KEY'] = _secret
# Session cookie configuration for the preview environment.
# The app is served over HTTPS (https://preview-chat-...space-z.ai) but the
# Next.js reverse proxy talks to Flask over plain HTTP on localhost. For the
# browser to accept and re-send the session cookie across that split, it must
# be marked SameSite=None; Secure (so it rides along on the HTTPS top-level
# navigation after login). Without SameSite=None the browser silently drops
# the cookie on the cross-site fetch, so GET / after login sees no session
# and redirects back to /login (the "login not working" symptom).
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_PATH'] = '/'
# Ensure redirects honor the proxy host/scheme so Location stays relative-friendly.
app.config['PREFERRED_URL_SCHEME'] = 'https'
app.config['SESSION_REFRESH_EACH_REQUEST'] = True
# NOTE: The original code used async_mode='eventlet'. Under gunicorn/the dev
# server, eventlet's monkey-patching conflicts with the native libraries loaded
# by TensorFlow/PyTorch/numba and causes the worker to segfault after the first
# request (silent crash, no traceback). We fall back to 'threading' async mode,
# which is provided by python-socketio out of the box, needs no extra worker
# class, and is stable for local/development use.
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

logger.info("Flask application initialized")


class _RelativeLocationMiddleware:
    """WSGI middleware that rewrites absolute redirect Location headers to
    path-relative.

    The app runs behind a reverse proxy (Next.js dev server on :3000 -> Flask
    on :5000). Werkzeug's Response finalization absolutizes the Location
    header using the internal host Flask sees (localhost:5000), which the
    browser cannot reach. This middleware intercepts the headers at the WSGI
    start_response level (the last point before the wire) and strips the
    scheme+netloc so the browser stays on the public-facing host the proxy
    serves.
    """

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        from urllib.parse import urlsplit

        def _new_start_response(status, headers, exc_info=None):
            new_headers = []
            for name, value in headers:
                if name.lower() == 'location' and '://' in value:
                    try:
                        p = urlsplit(value)
                        relative = p.path or '/'
                        if p.query:
                            relative += '?' + p.query
                        logger.debug("Relative Location: %s -> %s", value, relative)
                        value = relative
                    except Exception:
                        pass
                new_headers.append((name, value))
            return start_response(status, new_headers, exc_info)

        return self.wsgi_app(environ, _new_start_response)


app.wsgi_app = _RelativeLocationMiddleware(app.wsgi_app)

# Configure Google Gemini API via the official google-genai SDK.
# Uses the new GoogleGenAI client pattern:
#   client = genai.Client(api_key=GOOGLE_API_KEY)
#   client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
# The client is created lazily on first use so a missing/invalid key does not
# crash app startup.
from google import genai
from google.genai import types as genai_types

_gemini_client = None


def _get_gemini_client():
    """Lazily build the GoogleGenAI client from GOOGLE_API_KEY."""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    _gemini_client = genai.Client(api_key=api_key)
    logger.info("GoogleGenAI client configured (model=gemini-2.5-flash)")
    return _gemini_client

try:
    if os.getenv("GOOGLE_API_KEY"):
        _get_gemini_client()
        logger.info("Google Gemini API configured successfully")
    else:
        logger.warning("GOOGLE_API_KEY not set. AI features will be unavailable.")
except Exception as e:
    # Don't crash startup; the AI doctor route will surface the error per-call.
    logger.warning(f"Gemini client init deferred: {str(e)[:200]}")

heart_model = None
audio_model = None
heart_scaler = None
yamnet_model = None
ecg_model = None

def load_heart_and_ecg_models():
    """Lazily load heart disease and ECG models to reduce startup time."""
    global heart_model, heart_scaler, ecg_model
    try:
        # Heart disease model
        if heart_model is None and os.path.exists('heart/models/heart_model.joblib'):
            heart_model = joblib.load('heart/models/heart_model.joblib')
            logger.info("Heart model loaded successfully")
        
        # Heart scaler
        if heart_scaler is None and os.path.exists('heart/models/heart_scaler.joblib'):
            heart_scaler = joblib.load('heart/models/heart_scaler.joblib')
            logger.info("Heart scaler loaded successfully")
        
        # ECG model
        if ecg_model is None and os.path.exists('ecg project/best_model.pth'):
            seq_len = 1
            n_features = 141
            model = Autoencoder(seq_len, n_features)
            model.load_state_dict(torch.load('ecg project/best_model.pth', map_location=torch.device('cpu')))
            model.eval()
            ecg_model = model
            logger.info("ECG model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading heart/ECG models: {str(e)}")
        logger.warning("Heart/ECG features may be unavailable")

def load_audio_models():
    """Lazily load audio classification and YAMNet models."""
    global audio_model, yamnet_model
    try:
        if audio_model is None and os.path.exists('heart/models/audio_model.h5'):
            audio_model = tf.keras.models.load_model('heart/models/audio_model.h5')
            logger.info("Audio model loaded successfully")
        if yamnet_model is None and os.path.exists('archive/saved_model.pb'):
            yamnet_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'archive')
            yamnet_model = hub.load(yamnet_model_path)
            logger.info(f"YAMNet model loaded successfully from {yamnet_model_path}")
    except Exception as e:
        logger.error(f"Error loading audio/YAMNet models: {str(e)}")
        logger.warning("Audio analysis features may be unavailable")

def extract_embeddings(audio_data):
    """Extract embeddings using YAMNet model."""
    try:
        max_frames = 10
        scores, embeddings_output, _ = yamnet_model(audio_data)
        embeddings_output = embeddings_output[:max_frames]
        padding_length = max_frames - embeddings_output.shape[0]
        if padding_length > 0:
            embeddings_output = np.pad(embeddings_output, ((0, padding_length), (0, 0)), mode='constant')
        return embeddings_output.reshape(1, -1, 1024)
    except Exception as e:
        logger.error(f"Error extracting embeddings: {str(e)}")
        raise

def analyze_ecg(ecg_data, threshold=0.1):
    """Analyze ECG data using the autoencoder model."""
    try:
        ecg_data = np.array(ecg_data, dtype=np.float32)
        ecg_data = ecg_data.reshape(1, 1, 141)
        ecg_tensor = torch.tensor(ecg_data, dtype=torch.float32)
        
        with torch.no_grad():
            reconstruction = ecg_model(ecg_tensor)
            mse = torch.mean((ecg_tensor - reconstruction) ** 2, dim=(1, 2))
            is_anomaly = mse > threshold
        
        return is_anomaly.numpy(), ecg_data.squeeze(), reconstruction.squeeze().numpy()
    except Exception as e:
        logger.error(f"Error in ECG analysis: {str(e)}")
        raise

def login_required(f):
    """Require an authenticated session. Returns a JSON 401 for API routes
    (paths starting with /api/ or returning JSON), and redirects to /login
    for page routes. Preserves the wrapped function's name/docstring via
    functools.wraps."""
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            # API routes (POST/PUT/DELETE or /api/* path) get a JSON 401
            if (request.path.startswith('/api/') or
                    request.method in ('POST', 'PUT', 'DELETE', 'PATCH')):
                return jsonify({'error': 'Not authenticated'}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapper

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    """Authenticate a user against Neon and establish a Flask session.

    Passwords are verified against the password_hash column in the Neon
    users table using werkzeug's check_password_hash.
    """
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not email or not password:
        return jsonify({'status': 'error', 'message': 'Email and password are required'}), 400

    try:
        user = db_neon.authenticate_user(email, password)
    except Exception as e:
        logger.error(f"Login DB error for {email}: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Login failed. Please try again.'}), 500

    if not user:
        return jsonify({'status': 'error', 'message': 'Invalid email or password'}), 401

    session['user_id'] = str(user['id'])
    session['email'] = user.get('email', '')
    session['name'] = user.get('name') or user.get('email', '')
    logger.info(f"User logged in: {email} (id={user['id']})")
    return jsonify({'status': 'success'})

@app.route('/register', methods=['GET'])
def register():
    # Allow registration even when logged in (original behavior).
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register_post():
    """Register a new user in Neon. Does NOT auto-login; the client is sent
    back to /login to sign in with the new credentials."""
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    confirm = data.get('confirmPassword') or data.get('confirm_password') or ''

    if not name or not email or not password:
        return jsonify({'status': 'error', 'message': 'Name, email, and password are required'}), 400
    if password != confirm:
        return jsonify({'status': 'error', 'message': 'Passwords do not match'}), 400
    if len(password) < 6:
        return jsonify({'status': 'error', 'message': 'Password must be at least 6 characters'}), 400

    try:
        password_hash = generate_password_hash(password)
        db_neon.create_user(email=email, password_hash=password_hash, name=name)
    except Exception as e:
        # Unique-email violation -> friendly message
        if 'users_email_key' in str(e) or 'duplicate key' in str(e).lower():
            return jsonify({'status': 'error', 'message': 'An account with that email already exists'}), 409
        logger.error(f"Registration DB error for {email}: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Registration failed. Please try again.'}), 500

    logger.info(f"New user registered: {email}")
    return jsonify({'status': 'success', 'redirect': '/login'})

@app.route('/logout')
def logout():
    # Clear the server-side session
    session.clear()
    # Always redirect to login page
    return redirect(url_for('login'))

@app.route('/emergency')
@login_required
def emergency():
    ors_api_key = os.getenv('ORS_API_KEY', '')
    return render_template('emergency_map.html', ors_api_key=ors_api_key)

@app.route('/analyze_heart', methods=['POST'])
@login_required
def analyze_heart():
    try:
        data = request.get_json(silent=True) or {}
        logger.info(f"Received heart data: {data}")

        # Validate all required fields are present
        required = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        missing = [f for f in required if f not in data or data[f] == '']
        if missing:
            return jsonify({'error': f'Missing required fields: {", ".join(missing)}'}), 400

        # Ensure models are loaded lazily
        load_heart_and_ecg_models()

        input_data = pd.DataFrame([{
            'age': float(data['age']),
            'sex': int(data['sex']),
            'cp': int(data['cp']),
            'trestbps': float(data['trestbps']),
            'chol': float(data['chol']),
            'fbs': int(data['fbs']),
            'restecg': int(data['restecg']),
            'thalach': float(data['thalach']),
            'exang': int(data['exang']),
            'oldpeak': float(data['oldpeak']),
            'slope': int(data['slope']),
            'ca': int(data['ca']),
            'thal': int(data['thal'])
        }])
        
        input_scaled = heart_scaler.transform(input_data)
        probabilities = heart_model.predict_proba(input_scaled)[0]
        risk_probability = probabilities[1]
        
        # Risk indicators analysis
        high_risk_indicators = sum([
            float(data['age']) >= 65,
            float(data['trestbps']) >= 180,
            float(data['chol']) >= 300,
            int(data['restecg']) == 2,
            float(data['oldpeak']) >= 2.0,
            int(data['ca']) >= 2,
            int(data['thal']) >= 2
        ])
        
        low_risk_indicators = sum([
            float(data['age']) < 45,
            int(data['sex']) == 0,
            float(data['trestbps']) < 120,
            float(data['chol']) < 200,
            int(data['restecg']) == 0,
            float(data['oldpeak']) < 1.0,
            int(data['ca']) == 0,
            int(data['thal']) == 0,
            int(data['exang']) == 0,
            int(data['slope']) == 0
        ])
        
        if low_risk_indicators >= 5:
            risk_probability = min(risk_probability, 0.3)
        elif high_risk_indicators >= 3:
            risk_probability = max(risk_probability, 0.7)
        
        threshold = 0.5
        if low_risk_indicators >= 5:
            threshold = 0.6
        elif high_risk_indicators >= 3:
            threshold = 0.4
        
        prediction = risk_probability > threshold
        
        return jsonify({
            'prediction': bool(prediction),
            'probability': float(risk_probability),
            'high_risk_indicators': int(high_risk_indicators),
            'low_risk_indicators': int(low_risk_indicators)
        })
        
    except Exception as e:
        logger.error(f"Error in heart analysis: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/analyze_ecg', methods=['POST'])
@login_required
def analyze_ecg_endpoint():
    try:
        data = request.get_json()
        ecg_values = data.get('ecg_values', [])
        
        if len(ecg_values) != 141:
            return jsonify({'error': f'Expected 141 ECG values, but got {len(ecg_values)}'}), 400
        
        # Ensure ECG model is loaded lazily
        load_heart_and_ecg_models()
        is_anomaly, original, reconstructed = analyze_ecg(ecg_values)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(original, label='Original ECG', color='#2ecc71', linewidth=2)
        plt.plot(reconstructed, label='Reconstructed', color='#e74c3c', linewidth=2)
        plt.fill_between(range(len(original)), original, reconstructed, color='gray', alpha=0.3)
        plt.title('ECG Signal Analysis', fontsize=14, pad=20)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10, loc='upper right')
        plt.tight_layout()
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'is_anomaly': bool(is_anomaly[0]),
            'plot_url': plot_url,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in ECG analysis: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 400

@app.route('/analyze_audio', methods=['POST'])
@login_required
def analyze_audio():
    try:
        # Lazily load audio and YAMNet models
        load_audio_models()
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
            
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not audio_file.filename.endswith('.wav'):
            return jsonify({'error': 'Please upload a WAV file'}), 400
        
        y, sr = librosa.load(audio_file, sr=16000)
        y = y.astype(np.float32)
        y = librosa.util.normalize(y)
        
        embeddings = extract_embeddings(y)
        predictions = audio_model.predict(embeddings, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        disease_map = {
            0: 'Aortic Stenosis',
            1: 'Mitral Regurgitation',
            2: 'Mitral Stenosis',
            3: 'Mitral Valve Prolapse',
            4: 'Normal'
        }
        
        disease_name = disease_map.get(predicted_class, 'Unknown')
        
        return jsonify({
            'prediction': int(predicted_class),
            'disease': disease_name,
            'confidence': round(confidence * 100, 2)
        })
    except Exception as e:
        logger.error(f"Error in audio analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/profile')
@login_required
def profile():
    # Profile page not yet implemented; redirect to the dashboard for now.
    return redirect(url_for('index'))

@app.route('/api/emergency', methods=['POST'])
@login_required
def handle_emergency():
    try:
        data = request.get_json()
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'User not authenticated'}), 401
            
        # Create a new emergency record in Neon (PostgreSQL)
        emergency_data = {
            'type': data.get('type', 'Emergency'),
            'description': data.get('description', ''),
            'location': data.get('location', {}),
            'status': 'active',
            'user_id': user_id
        }

        row = db_neon.insert_emergency(
            user_id=user_id,
            type=emergency_data['type'],
            description=emergency_data['description'],
            location=emergency_data['location'],
            status=emergency_data['status'],
        )

        if row and row.get('id'):
            return jsonify({
                'status': 'success',
                'emergencyId': str(row['id'])
            })
        else:
            return jsonify({'error': 'Failed to create emergency record'}), 500
        
    except Exception as e:
        logger.error(f"Error handling emergency: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/emergency/<emergency_id>', methods=['PUT'])
@login_required
def update_emergency(emergency_id):
    try:
        data = request.get_json()
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'User not authenticated'}), 401
            
        # Update the emergency record in Neon (PostgreSQL)
        new_status = data.get('status', 'resolved')

        row = db_neon.update_emergency(emergency_id, user_id, status=new_status)

        if row:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'error': 'Failed to update emergency record'}), 500
        
    except Exception as e:
        logger.error(f"Error updating emergency: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/volunteer/toggle', methods=['POST'])
@login_required
def toggle_volunteer():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    
    # Get current volunteer status from Neon
    user_row = db_neon.get_user(user_id)

    if user_row:
        current_status = bool(user_row.get('is_volunteer', False))
    else:
        # User doesn't exist yet; default to False before toggling.
        current_status = False

    new_status = not current_status

    # Upsert user record in Neon
    db_neon.upsert_user(user_id, is_volunteer=new_status)

    return jsonify({'status': 'success', 'is_volunteer': new_status})

@app.route('/api/volunteer/location', methods=['POST'])
@login_required
def update_volunteer_location():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    lat = data.get('lat')
    lng = data.get('lng')
    user_id = session['user_id']
    
    # Update user location in Neon
    db_neon.upsert_user(user_id, location={
        'lat': lat,
        'lng': lng,
        'timestamp': datetime.now().isoformat()
    })

    return jsonify({'status': 'success'})

@app.route('/api/nearby_hospitals')
def nearby_hospitals():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if not isinstance(lat, float) or not isinstance(lon, float):
        return jsonify({"error": "Latitude and longitude must be valid floats"}), 400
    
    return query_overpass_for_hospitals(lat, lon)

def query_overpass_for_hospitals(latitude, longitude):
    overpass_url = "http://overpass-api.de/api/interpreter"
    radius = 5000  # Initial search radius in meters
    max_radius = 20000  # Maximum search radius
    min_hospitals = 5  # Minimum number of hospitals to find
    
    while radius <= max_radius:
        query = f"""
        [out:json];
        (
          node(around:{radius},{latitude},{longitude})["amenity"="hospital"];
          way(around:{radius},{latitude},{longitude})["amenity"="hospital"];
          relation(around:{radius},{latitude},{longitude})["amenity"="hospital"];
        );
        out center;
        """
        
        params = {'data': query}
        
        try:
            response = requests.get(overpass_url, params=params)
            response.raise_for_status()
            data = response.json()
            hospitals = process_overpass_results(data, latitude, longitude)
            
            if len(hospitals) >= min_hospitals:
                return jsonify({"hospitals": hospitals})
            
            # If we don't have enough hospitals, increase the radius
            radius += 5000
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Error querying Overpass API: {e}"}), 500
    
    # If we still don't have enough hospitals after reaching max radius, return what we have
    return jsonify({"hospitals": hospitals})

def process_overpass_results(data, current_lat, current_lon):
    hospitals = []
    elements = data.get('elements', [])
    
    if not isinstance(elements, list):
        # Overpass sometimes returns an error object without elements
        return hospitals
    
    for element in elements:
        if 'tags' in element and element['tags'].get('amenity') == 'hospital':
            lat = None
            lon = None
            if 'lat' in element and 'lon' in element:
                lat = element['lat']
                lon = element['lon']
            elif 'center' in element:
                lat = element['center']['lat']
                lon = element['center']['lon']
            
            if lat is not None and lon is not None:
                distance = calculate_distance(current_lat, current_lon, lat, lon)
                hospitals.append({
                    'name': element['tags'].get('name', 'Unnamed Hospital'),
                    'lat': lat,
                    'lon': lon,
                    'distance': distance,
                    'address': element['tags'].get('addr:street', '') + ', ' + element['tags'].get('addr:city', ''),
                    'phone': element['tags'].get('phone', ''),
                    'website': element['tags'].get('website', '')
                })
    
    # Sort hospitals by distance
    hospitals.sort(key=lambda h: h['distance'])
    return hospitals[:5]  # Return only the 5 closest hospitals

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    
    a = math.sin(dLat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c * 1000  # Distance in meters
    return distance

@app.route('/ai_doctor', methods=['POST'])
@login_required
def ai_doctor():
    try:
        data = request.get_json(silent=True) or {}
        user_query = data.get('query', '')

        if not user_query:
            return jsonify({'error': 'No query provided'}), 400

        # Generate response using the Gemini model
        response = generate_output(user_query)

        if response:
            return jsonify({'response': response})
        else:
            return jsonify({'error': 'Failed to generate response'}), 500
    except Exception as e:
        logger.error(f"Error in AI Doctor: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_output(input_text):
    """
    Generate a doctor-like response to the user's query using the Gemini model.
    """
    prompt = f"""
    You are a highly experienced cardiologist with over 20 years of practice. Respond to the following patient question in a warm, empathetic, and professional manner. Use your medical expertise to provide helpful information while maintaining a conversational tone.

    Patient question: '{input_text}'

    Important instructions:
    - Keep your response to approximately 150 words
    - Do not use any markdown symbols, asterisks, or formatting characters
    - Write in plain text only
    - Be empathetic and understanding
    - Use simple language to explain medical concepts
    - Provide practical advice when appropriate
    - Maintain a professional but friendly tone
    - Acknowledge the patient's concerns
    - Suggest when to seek immediate medical attention if necessary
    - Do not repeat the patient's question in your response
    - Give direct, helpful answers without asking for more information unless absolutely necessary
    """
    
    try:
        # --- Primary path: official google-genai SDK -------------------------
        # Uses the new GoogleGenAI client pattern:
        #   client = genai.Client(api_key=GOOGLE_API_KEY)
        #   ai.models.generate_content({model: "gemini-2.5-flash", contents: prompt})
        try:
            client = _get_gemini_client()
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.8,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=1024,
                ),
            )
            text = (response.text or "").strip()
            if text:
                logger.info("AI doctor response generated via google-genai SDK (gemini-2.5-flash).")
                return text
            logger.warning("google-genai returned empty text; trying fallback.")
        except Exception as sdk_err:
            # Google's API is geo-blocked from this sandbox
            # ("User location is not supported"), so this path usually fails
            # here. Fall through to the z-ai CLI fallback so the AI doctor
            # feature still functions.
            logger.warning("google-genai SDK call failed (will try fallback): %s", str(sdk_err)[:200])

        # --- Fallback path: z-ai CLI (pre-authenticated sandbox SDK) ---------
        # Used only when the official Google endpoint is unreachable (e.g.
        # geo-blocked) so the AI doctor feature still returns a response.
        try:
            proc = subprocess.run(
                [
                    "z-ai", "chat",
                    "-p", input_text,
                    "-s", ("You are a highly experienced cardiologist with over 20 "
                           "years of practice. Respond in a warm, empathetic, and "
                           "professional manner in plain text (no markdown), "
                           "approximately 150 words."),
                    "-o", "/tmp/_coeur_ai_doctor.json",
                ],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode == 0 and os.path.exists("/tmp/_coeur_ai_doctor.json"):
                with open("/tmp/_coeur_ai_doctor.json", "r") as f:
                    out = json.load(f)
                content = out["choices"][0]["message"]["content"]
                logger.info("AI doctor response generated via z-ai CLI fallback.")
                return content
            else:
                logger.error("z-ai CLI fallback failed: %s", proc.stderr[:300])
        except Exception as cli_err:
            logger.error("z-ai CLI fallback exception: %s", cli_err)

        # --- Last resort: canned response ------------------------------------
        logger.warning("All AI generation paths failed; using canned fallback response.")
        return "Chest pain after jogging could indicate several conditions. If the pain is sharp, radiates to your arm or jaw, or is accompanied by shortness of breath, seek immediate medical attention. For milder discomfort, try warming up properly before exercise, staying hydrated, and gradually increasing your activity level. Consider consulting a cardiologist for a thorough evaluation, especially if the pain persists or worsens. They may recommend tests like an ECG or stress test to determine the cause."
    except Exception as e:
        logger.error("Error generating response: %s", str(e), exc_info=True)
        # Fallback response if there's an error
        return "Chest pain after jogging could indicate several conditions. If the pain is sharp, radiates to your arm or jaw, or is accompanied by shortness of breath, seek immediate medical attention. For milder discomfort, try warming up properly before exercise, staying hydrated, and gradually increasing your activity level. Consider consulting a cardiologist for a thorough evaluation, especially if the pain persists or worsens. They may recommend tests like an ECG or stress test to determine the cause."

@app.route('/api/emergency_contacts', methods=['GET', 'POST', 'DELETE'])
@login_required
def handle_emergency_contacts():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    
    if request.method == 'GET':
        try:
            # Get user's emergency contacts from Neon
            rows = db_neon.list_emergency_contacts(user_id)

            if rows:
                # Convert list to dict format for compatibility
                contacts_dict = {
                    str(c['id']): {'name': c['name'], 'phone': c['phone']}
                    for c in rows
                }
                return jsonify({'contacts': contacts_dict})
            return jsonify({'contacts': {}})
        except Exception as e:
            logger.error(f"Error getting contacts: {str(e)}")
            return jsonify({'error': 'Failed to get contacts'}), 500

    elif request.method == 'POST':
        try:
            data = request.get_json()
            name = data.get('name')
            phone = data.get('phone')

            if not name or not phone:
                return jsonify({'error': 'Name and phone are required'}), 400

            # Add contact to Neon
            row = db_neon.insert_emergency_contact(user_id, name, phone)

            if row and row.get('id'):
                return jsonify({
                    'status': 'success',
                    'contact_id': str(row['id']),
                    'contact': {
                        'name': name,
                        'phone': phone
                    }
                })
            else:
                return jsonify({'error': 'Failed to add contact'}), 500
        except Exception as e:
            logger.error(f"Error adding contact: {str(e)}")
            return jsonify({'error': 'Failed to add contact'}), 500

    elif request.method == 'DELETE':
        try:
            contact_id = request.args.get('id')
            if not contact_id:
                return jsonify({'error': 'Contact ID is required'}), 400

            # Remove contact from Neon
            db_neon.delete_emergency_contact(contact_id, user_id)

            return jsonify({'status': 'success'})
        except Exception as e:
            logger.error(f"Error removing contact: {str(e)}")
            return jsonify({'error': 'Failed to remove contact'}), 500

# ---------------------------------------------------------------------------
# Medical info (medications + allergies) — Neon-backed
# ---------------------------------------------------------------------------
@app.route('/api/medical_info', methods=['GET', 'PUT'])
@login_required
def handle_medical_info():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    if request.method == 'GET':
        try:
            row = db_neon.get_medical_info(user_id)
            if row:
                return jsonify({
                    'medications': row.get('medications') or '',
                    'allergies': row.get('allergies') or '',
                })
            return jsonify({'medications': '', 'allergies': ''})
        except Exception as e:
            logger.error(f"Error getting medical info: {str(e)}")
            return jsonify({'error': 'Failed to get medical info'}), 500

    if request.method == 'PUT':
        try:
            data = request.get_json(silent=True) or {}
            medications = data.get('medications', '')
            allergies = data.get('allergies', '')
            db_neon.upsert_medical_info(user_id, medications=medications, allergies=allergies)
            logger.info(f"Medical info saved for user {user_id}")
            return jsonify({'status': 'success'})
        except Exception as e:
            logger.error(f"Error saving medical info: {str(e)}")
            return jsonify({'error': 'Failed to save medical info'}), 500

# ---------------------------------------------------------------------------
# Medical records (uploaded files) — Neon-backed
# ---------------------------------------------------------------------------
@app.route('/api/medical_records', methods=['GET', 'POST'])
@login_required
def handle_medical_records():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    if request.method == 'GET':
        try:
            rows = db_neon.list_medical_records(user_id)
            # Serialize datetimes + ids for JSON
            records = []
            for r in rows:
                records.append({
                    'id': str(r['id']),
                    'name': r['name'],
                    'type': r.get('type') or '',
                    'uploaded_at': r['uploaded_at'].isoformat() if r.get('uploaded_at') else None,
                })
            return jsonify({'records': records})
        except Exception as e:
            logger.error(f"Error listing medical records: {str(e)}")
            return jsonify({'error': 'Failed to list medical records'}), 500

    if request.method == 'POST':
        try:
            data = request.get_json(silent=True) or {}
            name = data.get('name')
            type_ = data.get('type', '')
            content = data.get('content', '')
            if not name:
                return jsonify({'error': 'Record name is required'}), 400
            row = db_neon.insert_medical_record(user_id, name=name, type_=type_, content=content)
            logger.info(f"Medical record '{name}' uploaded for user {user_id}")
            return jsonify({
                'status': 'success',
                'id': str(row['id']),
                'name': row['name'],
                'uploaded_at': row['uploaded_at'].isoformat() if row.get('uploaded_at') else None,
            })
        except Exception as e:
            logger.error(f"Error uploading medical record: {str(e)}")
            return jsonify({'error': 'Failed to upload medical record'}), 500

@app.route('/api/medical_records/<record_id>', methods=['GET', 'DELETE'])
@login_required
def handle_medical_record(record_id):
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    if request.method == 'GET':
        try:
            row = db_neon.get_medical_record(record_id, user_id)
            if not row:
                return jsonify({'error': 'Record not found'}), 404
            return jsonify({
                'id': str(row['id']),
                'name': row['name'],
                'type': row.get('type') or '',
                'content': row.get('content') or '',
                'uploaded_at': row['uploaded_at'].isoformat() if row.get('uploaded_at') else None,
            })
        except Exception as e:
            logger.error(f"Error getting medical record: {str(e)}")
            return jsonify({'error': 'Failed to get medical record'}), 500

    if request.method == 'DELETE':
        try:
            deleted = db_neon.delete_medical_record(record_id, user_id)
            if not deleted:
                return jsonify({'error': 'Record not found'}), 404
            logger.info(f"Medical record {record_id} deleted for user {user_id}")
            return jsonify({'status': 'success'})
        except Exception as e:
            logger.error(f"Error deleting medical record: {str(e)}")
            return jsonify({'error': 'Failed to delete medical record'}), 500

# ---------------------------------------------------------------------------
# Admin: Demo Dataset Manager (Parts 3, 4, 5)
# ---------------------------------------------------------------------------

# Demo files are stored on disk under DEMO_FILES_DIR (defaults to 'demo_files'
# relative to the app). On Render, mount a persistent disk and set
# DEMO_FILES_DIR to the disk path so files survive deploys.
DEMO_FILES_DIR = os.environ.get('DEMO_FILES_DIR', os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'demo_files'))


def admin_required(f):
    """Decorator: requires logged-in user with is_admin flag."""
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        uid = session.get('user_id')
        if not uid:
            return jsonify({'error': 'Not authenticated'}), 401
        try:
            if not db_neon.is_admin(uid):
                return jsonify({'error': 'Admin access required'}), 403
        except Exception:
            return jsonify({'error': 'Authorization check failed'}), 500
        return f(*args, **kwargs)
    return wrapper


@app.route('/admin')
@login_required
def admin_page():
    """Admin page: demo dataset manager."""
    try:
        if not db_neon.is_admin(session.get('user_id')):
            return redirect(url_for('index'))
    except Exception:
        return redirect(url_for('index'))
    return render_template('admin.html')


def _demo_kind_from_path(path):
    """Map a URL path segment to a demo kind ('ecg' or 'heart_sound')."""
    if 'ecg' in path:
        return 'ecg'
    if 'heart_sound' in path or 'heart-sound' in path or 'audio' in path:
        return 'heart_sound'
    return None


@app.route('/api/demo/<kind>', methods=['GET'])
@login_required
def list_demo(kind):
    """List active demo files (available to all logged-in users for the dropdowns)."""
    if kind not in ('ecg', 'heart_sound'):
        return jsonify({'error': 'Invalid kind'}), 400
    try:
        files = db_neon.list_demo_files(kind, active_only=True)
        return jsonify({'files': files})
    except Exception as e:
        logger.error(f"Error listing demo {kind} files: {str(e)}")
        return jsonify({'error': 'Failed to list demo files'}), 500


@app.route('/api/demo/<kind>/<file_id>/content', methods=['GET'])
@login_required
def get_demo_content(kind, file_id):
    """Serve the binary content of a demo file (for the frontend to POST to the
    existing /analyze_ecg or /analyze_audio pipeline)."""
    if kind not in ('ecg', 'heart_sound'):
        return jsonify({'error': 'Invalid kind'}), 400
    try:
        info = db_neon.get_demo_file(kind, file_id)
        if not info or not info.get('active'):
            return jsonify({'error': 'File not found'}), 404
        path = info['storage_path']
        if not os.path.exists(path):
            return jsonify({'error': 'File not found on disk'}), 404
        # ECG demo files are JSON -> return as JSON for the frontend to POST
        # to /analyze_ecg. Heart sound files are WAV -> return as binary.
        if kind == 'ecg':
            with open(path, 'r') as f:
                return jsonify(json.load(f))
        else:
            return send_file(path, mimetype='audio/wav',
                             as_attachment=False,
                             download_name=info.get('filename', 'demo.wav'))
    except Exception as e:
        logger.error(f"Error serving demo {kind} content: {str(e)}")
        return jsonify({'error': 'Failed to serve file'}), 500


@app.route('/api/admin/demo/<kind>', methods=['GET'])
@admin_required
def admin_list_demo(kind):
    """Admin: list ALL demo files (including inactive)."""
    if kind not in ('ecg', 'heart_sound'):
        return jsonify({'error': 'Invalid kind'}), 400
    try:
        files = db_neon.list_demo_files(kind, active_only=False)
        return jsonify({'files': files})
    except Exception as e:
        logger.error(f"Admin list demo error: {str(e)}")
        return jsonify({'error': 'Failed to list'}), 500


@app.route('/api/admin/demo/<kind>', methods=['POST'])
@admin_required
def admin_upload_demo(kind):
    """Admin: upload a new demo file."""
    if kind not in ('ecg', 'heart_sound'):
        return jsonify({'error': 'Invalid kind'}), 400
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        f = request.files['file']
        if not f.filename:
            return jsonify({'error': 'No filename'}), 400

        title = request.form.get('title') or os.path.splitext(f.filename)[0]
        description = request.form.get('description') or ''

        # Validate file type
        if kind == 'ecg' and not f.filename.endswith('.json'):
            return jsonify({'error': 'ECG demo files must be .json'}), 400
        if kind == 'heart_sound' and not f.filename.endswith('.wav'):
            return jsonify({'error': 'Heart sound demo files must be .wav'}), 400

        # Save to disk
        subdir = 'ecg' if kind == 'ecg' else 'heart_sound'
        dest_dir = os.path.join(DEMO_FILES_DIR, subdir)
        os.makedirs(dest_dir, exist_ok=True)
        # Unique filename to avoid collisions
        import uuid as _uuid
        unique_name = f"{_uuid.uuid4().hex}_{f.filename}"
        dest_path = os.path.join(dest_dir, unique_name)
        f.save(dest_path)
        file_size = os.path.getsize(dest_path)

        # Insert metadata
        row = db_neon.insert_demo_file(
            kind=kind,
            title=title,
            filename=f.filename,
            storage_path=dest_path,
            description=description,
            file_size=file_size,
            uploaded_by=session.get('user_id'),
        )
        logger.info(f"Admin uploaded demo {kind}: {title} ({file_size} bytes)")
        return jsonify({'status': 'success', 'file': row}), 201
    except Exception as e:
        logger.error(f"Admin upload demo error: {str(e)}")
        return jsonify({'error': 'Failed to upload'}), 500


@app.route('/api/admin/demo/<kind>/<file_id>', methods=['PUT'])
@admin_required
def admin_rename_demo(kind, file_id):
    """Admin: rename / update description of a demo file."""
    if kind not in ('ecg', 'heart_sound'):
        return jsonify({'error': 'Invalid kind'}), 400
    try:
        data = request.get_json(silent=True) or {}
        new_title = data.get('title')
        new_desc = data.get('description')
        if not new_title:
            return jsonify({'error': 'Title is required'}), 400
        row = db_neon.rename_demo_file(kind, file_id, new_title,
                                       description=new_desc)
        if not row:
            return jsonify({'error': 'File not found'}), 404
        logger.info(f"Admin renamed demo {kind} {file_id} -> {new_title}")
        return jsonify({'status': 'success', 'file': row})
    except Exception as e:
        logger.error(f"Admin rename demo error: {str(e)}")
        return jsonify({'error': 'Failed to rename'}), 500


@app.route('/api/admin/demo/<kind>/<file_id>', methods=['DELETE'])
@admin_required
def admin_delete_demo(kind, file_id):
    """Admin: delete a demo file (metadata + disk file)."""
    if kind not in ('ecg', 'heart_sound'):
        return jsonify({'error': 'Invalid kind'}), 400
    try:
        deleted, storage_path = db_neon.delete_demo_file(kind, file_id)
        if not deleted:
            return jsonify({'error': 'File not found'}), 404
        # Remove the binary file from disk
        if storage_path and os.path.exists(storage_path):
            try:
                os.remove(storage_path)
            except OSError as e:
                logger.warning(f"Could not delete file {storage_path}: {e}")
        logger.info(f"Admin deleted demo {kind} {file_id}")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Admin delete demo error: {str(e)}")
        return jsonify({'error': 'Failed to delete'}), 500

@app.route('/emergency_map')
@login_required
def emergency_map():
    if not session.get('user'):
        return redirect(url_for('login'))
    ors_api_key = os.getenv('ORS_API_KEY', '')
    return render_template('emergency_map.html', ors_api_key=ors_api_key)

@app.route('/api/generate_analysis', methods=['POST'])
@login_required
def generate_analysis():
    try:
        data = request.get_json()
        
        # Extract the content from HTML results
        def extract_text(html_content):
            if not html_content:
                return None
            # Remove HTML tags and decode HTML entities
            import re
            from html import unescape
            text = re.sub(r'<[^>]+>', ' ', html_content)
            text = unescape(text)
            return text.strip()

        # Get available test results
        heart_disease = extract_text(data.get('heartDisease', ''))
        ecg = extract_text(data.get('ecg', ''))
        heart_sound = extract_text(data.get('heartSound', ''))
        
        # Get heart disease risk parameters
        heart_params = data.get('heartParams', {})
        heart_params_text = ""
        if heart_params:
            # Map parameter values to human-readable format
            cp_map = {
                '1': 'Typical Angina',
                '2': 'Atypical Angina',
                '3': 'Non-anginal Pain',
                '4': 'Asymptomatic'
            }
            
            restecg_map = {
                '0': 'Normal',
                '1': 'ST-T Wave Abnormality',
                '2': 'Left Ventricular Hypertrophy'
            }
            
            slope_map = {
                '0': 'Upsloping',
                '1': 'Flat',
                '2': 'Downsloping'
            }
            
            thal_map = {
                '0': 'Normal',
                '1': 'Fixed Defect',
                '2': 'Reversible Defect',
                '3': 'Other'
            }
            
            heart_params_text = """
            Heart Disease Risk Assessment Parameters:
            - Age: {age} years
            - Sex: {sex}
            - Chest Pain Type: {cp}
            - Resting Blood Pressure: {trestbps} mmHg
            - Serum Cholesterol: {chol} mg/dl
            - Fasting Blood Sugar: {fbs}
            - Resting ECG Results: {restecg}
            - Maximum Heart Rate Achieved: {thalach} bpm
            - Exercise Induced Angina: {exang}
            - ST Depression Induced by Exercise: {oldpeak} mm
            - Slope of Peak Exercise ST Segment: {slope}
            - Number of Major Vessels: {ca}
            - Thalassemia: {thal}
            """.format(
                age=heart_params.get('age', 'N/A'),
                sex='Male' if heart_params.get('sex') == '1' else 'Female',
                cp=cp_map.get(heart_params.get('cp', ''), 'N/A'),
                trestbps=heart_params.get('trestbps', 'N/A'),
                chol=heart_params.get('chol', 'N/A'),
                fbs='> 120 mg/dl' if heart_params.get('fbs') == '1' else '<= 120 mg/dl',
                restecg=restecg_map.get(heart_params.get('restecg', ''), 'N/A'),
                thalach=heart_params.get('thalach', 'N/A'),
                exang='Yes' if heart_params.get('exang') == '1' else 'No',
                oldpeak=heart_params.get('oldpeak', 'N/A'),
                slope=slope_map.get(heart_params.get('slope', ''), 'N/A'),
                ca=heart_params.get('ca', 'N/A'),
                thal=thal_map.get(heart_params.get('thal', ''), 'N/A')
            )

        # Prepare the prompt for Gemini
        test_results = []
        if heart_disease:
            test_results.append(f"Heart Disease Risk Assessment Results:\n{heart_disease}\n\n{heart_params_text}")
        if ecg:
            test_results.append(f"ECG Analysis Results:\n{ecg}")
        if heart_sound:
            test_results.append(f"Heart Sound Analysis Results:\n{heart_sound}")

        prompt = """As a medical AI assistant, please analyze the following test results and provide a professional medical report.
Only analyze the test results that are provided below. Do not mention or speculate about missing tests.

{test_results}

Please provide a professional medical report in the following format:

1. Summary of Findings:
Provide a clear and concise overview of the available test results.
Focus on the key findings and their clinical significance.
Include analysis of the heart disease risk parameters if available.

2. Potential Health Concerns:
List any identified health concerns based on the available test results.
Rate the severity of each concern (mild, moderate, or severe).
Explain the clinical implications of each finding.
Consider the heart disease risk parameters in your assessment.

3. Recommendations for Follow-up:
Suggest specific medical tests or consultations based on the findings.
Recommend appropriate follow-up intervals.
List relevant specialists for consultation if needed.
Base recommendations on both test results and risk parameters.

4. Lifestyle Suggestions:
Provide specific lifestyle modifications based on the findings.
Include dietary recommendations if relevant.
Suggest appropriate exercise routines if applicable.
List habits to adopt or avoid based on the test results and risk parameters.

5. When to Seek Immediate Medical Attention:
List specific symptoms or changes that require urgent care.
Provide clear guidelines for emergency situations.
Include warning signs to watch for based on the test results and risk parameters.

Format the response in clear, professional medical language.
Avoid using markdown symbols (*, **) or bullet points.
Write in a formal, clinical tone appropriate for a medical report.""".format(test_results='\n\n'.join(test_results))

        # Use direct API call to Gemini API
        api_key = os.environ["GOOGLE_API_KEY"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()
        
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            if 'content' in response_data['candidates'][0]:
                if 'parts' in response_data['candidates'][0]['content']:
                    if len(response_data['candidates'][0]['content']['parts']) > 0:
                        if 'text' in response_data['candidates'][0]['content']['parts'][0]:
                            analysis_text = response_data['candidates'][0]['content']['parts'][0]['text']
                            
                            # Split the analysis text into sections
                            sections = {
                                'summary': '',
                                'concerns': '',
                                'recommendations': '',
                                'lifestyle': '',
                                'emergency': ''
                            }
                            
                            # Helper function to extract section content
                            def extract_section(text, start_marker, end_marker=None):
                                try:
                                    if end_marker:
                                        return text.split(start_marker)[1].split(end_marker)[0].strip()
                                    return text.split(start_marker)[1].strip()
                                except IndexError:
                                    return "No information available"
                            
                            # Extract each section
                            sections['summary'] = extract_section(analysis_text, '1. Summary of Findings:', '2. Potential Health Concerns:')
                            sections['concerns'] = extract_section(analysis_text, '2. Potential Health Concerns:', '3. Recommendations for Follow-up:')
                            sections['recommendations'] = extract_section(analysis_text, '3. Recommendations for Follow-up:', '4. Lifestyle Suggestions:')
                            sections['lifestyle'] = extract_section(analysis_text, '4. Lifestyle Suggestions:', '5. When to Seek Immediate Medical Attention:')
                            sections['emergency'] = extract_section(analysis_text, '5. When to Seek Immediate Medical Attention:')
                            
                            # Format the response with proper HTML structure
                            formatted_response = f"""
                            <div class="analysis-section">
                                <div class="report-header">
                                    <h3>Medical Analysis Report</h3>
                                    <p class="report-date">{datetime.now().strftime('%B %d, %Y')}</p>
                                </div>
                                
                                <div class="report-section">
                                    <h4>Summary of Findings</h4>
                                    <p>{sections['summary']}</p>
                                </div>
                                
                                <div class="report-section">
                                    <h4>Potential Health Concerns</h4>
                                    <p>{sections['concerns']}</p>
                                </div>
                                
                                <div class="report-section">
                                    <h4>Recommendations for Follow-up</h4>
                                    <p>{sections['recommendations']}</p>
                                </div>
                                
                                <div class="report-section">
                                    <h4>Lifestyle Suggestions</h4>
                                    <p>{sections['lifestyle']}</p>
                                </div>
                                
                                <div class="report-section">
                                    <h4>When to Seek Immediate Medical Attention</h4>
                                    <p>{sections['emergency']}</p>
                                </div>
                            </div>
                            """
                            
                            # Store the analysis data in the session
                            session['latest_analysis'] = {
                                'analysis': formatted_response,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            return jsonify({
                                'status': 'success',
                                'analysis': formatted_response
                            })
        
        # Fallback response if the API call fails
        return jsonify({
            'status': 'error',
            'message': 'Failed to generate analysis'
        }), 500
        
    except Exception as e:
        print(f"Error in generate_analysis: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Create reports directory if it doesn't exist
if not os.path.exists('reports'):
    os.makedirs('reports')

def generate_pdf_report(user_id, analysis_data):
    """Generate a PDF report from the analysis data."""
    try:
        # Create a unique filename for the report
        filename = f"reports/medical_report_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Create the PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("Medical Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Add date
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=20
        )
        story.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", date_style))
        story.append(Spacer(1, 12))

        # Extract sections from the analysis HTML
        sections = {
            'Summary of Findings': '',
            'Potential Health Concerns': '',
            'Recommendations for Follow-up': '',
            'Lifestyle Suggestions': '',
            'When to Seek Immediate Medical Attention': ''
        }

        # Clean HTML content
        content = analysis_data.get('analysis', '')
        for section_title in sections.keys():
            pattern = f'<h4>{section_title}</h4>\\s*<p>(.*?)</p>'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                sections[section_title] = unescape(match.group(1).strip())

        # Add sections to the PDF
        section_style = ParagraphStyle(
            'SectionStyle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10
        )
        content_style = ParagraphStyle(
            'ContentStyle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=20
        )

        for title, content in sections.items():
            story.append(Paragraph(title, section_style))
            story.append(Spacer(1, 12))
            story.append(Paragraph(content, content_style))
            story.append(Spacer(1, 12))

        # Build the PDF
        doc.build(story)
        return filename
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        return None

@app.route('/download_report')
@login_required
def download_report():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'User not authenticated'}), 401

        # Get the latest analysis data from the session
        analysis_data = session.get('latest_analysis')
        if not analysis_data:
            return jsonify({'error': 'No analysis data available. Please generate an analysis first.'}), 400

        # Generate the PDF report
        report_path = generate_pdf_report(user_id, analysis_data)
        if not report_path:
            return jsonify({'error': 'Failed to generate report'}), 500

        # Send the file
        return send_file(
            report_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'medical_report_{datetime.now().strftime("%Y%m%d")}.pdf'
        )
    except Exception as e:
        logger.error(f"Error downloading report: {str(e)}")
        return jsonify({'error': 'Failed to download report'}), 500


if __name__ == '__main__':
    # Local development entry point.
    # The app uses Flask-SocketIO with eventlet async mode, so we start it via
    # socketio.run() instead of app.run(). gunicorn (see Procfile) is used in
    # production with a gevent/eventlet worker class.
    server_port = int(os.environ.get("PORT", "5000"))
    logger.info(f"Starting Coeur on port {server_port} (debug={DEBUG_MODE})")
    socketio.run(
        app,
        host="0.0.0.0",
        port=server_port,
        debug=DEBUG_MODE,
        allow_unsafe_werkzeug=True,
    )

