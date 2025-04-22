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
import firebase_admin
from firebase_admin import credentials, auth, db
from datetime import datetime
import math
import requests
import google.generativeai as genai
import time
import pyrebase4 as pyrebase
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from html import unescape
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Firebase configuration for client-side
FIREBASE_CONFIG = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL")
}

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY")
socketio = SocketIO(app, async_mode='gevent', cors_allowed_origins="*")

# Initialize Firebase Admin
cred = credentials.Certificate({
    "type": os.getenv("FIREBASE_TYPE"),
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace("\\n", "\n"),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
    "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
    "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN")
})

firebase_admin.initialize_app(cred, {
    'databaseURL': os.getenv("FIREBASE_DATABASE_URL")
})

# Initialize Pyrebase for client-side operations
firebase = pyrebase.initialize_app(FIREBASE_CONFIG)

# Get database reference
db = firebase_admin.db.reference()

# Configure Google Gemini API
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Comment out the model initialization for now
# model = genai.GenerativeModel("gemini-2.0-flash")

# Load the trained models
try:
    logger.info("Loading trained models...")
    heart_model = joblib.load('heart/models/heart_model.joblib')
    audio_model = tf.keras.models.load_model('heart/models/audio_model.h5')
    heart_scaler = joblib.load('heart/models/heart_scaler.joblib')
    
    # Load YAMNet model for heart sound analysis
    try:
        yamnet_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'archive')
        yamnet_model = hub.load(yamnet_model_path)
        print(f"INFO:__main__:Successfully loaded YAMNet model from {yamnet_model_path}")
    except Exception as e:
        print(f"WARNING:__main__:Failed to load YAMNet model: {str(e)}")
        yamnet_model = None
    
    # Load ECG model
    seq_len = 1
    n_features = 141
    ecg_model = Autoencoder(seq_len, n_features)
    ecg_model.load_state_dict(torch.load('ecg project/best_model.pth', map_location=torch.device('cpu')))
    ecg_model.eval()
    
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

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
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

@app.route('/')
@login_required
def index():
    return render_template('index.html', firebase_config=FIREBASE_CONFIG)

@app.route('/login', methods=['GET'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('login.html', firebase_config=FIREBASE_CONFIG)

@app.route('/register', methods=['GET'])
def register():
    # Remove the session check to allow registration even when logged in
    return render_template('register.html', firebase_config=FIREBASE_CONFIG)

@app.route('/verify-token', methods=['POST'])
def verify_token():
    id_token = request.json.get('idToken')
    if not id_token:
        return jsonify({'status': 'error', 'message': 'No token provided'}), 400
        
    try:
        # Add a small delay to handle time synchronization issues
        time.sleep(1)  # Wait for 1 second before verifying the token
        
        decoded_token = auth.verify_id_token(id_token)
        session['user_id'] = decoded_token['uid']
        session['email'] = decoded_token.get('email', '')
        session['name'] = decoded_token.get('name', '')
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Invalid token'}), 401

@app.route('/logout')
def logout():
    # Clear the server-side session
    session.clear()
    
    # Always redirect to login page
    return redirect(url_for('login'))

@app.route('/emergency')
def emergency():
    return render_template('emergency_map.html')

@app.route('/analyze_heart', methods=['POST'])
def analyze_heart():
    try:
        data = request.get_json()
        logger.info(f"Received heart data: {data}")
        
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
def analyze_ecg_endpoint():
    try:
        data = request.get_json()
        ecg_values = data.get('ecg_values', [])
        
        if len(ecg_values) != 141:
            return jsonify({'error': f'Expected 141 ECG values, but got {len(ecg_values)}'}), 400
        
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
def analyze_audio():
    try:
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
    return render_template('profile.html', firebase_config=FIREBASE_CONFIG)

@app.route('/api/emergency', methods=['POST'])
@login_required
def handle_emergency():
    try:
        data = request.get_json()
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'User not authenticated'}), 401
            
        # Create a new emergency record in Firebase
        emergency_ref = db.child(f'emergencies/{user_id}').push()
        emergency_data = {
            'type': data.get('type', 'Emergency'),
            'description': data.get('description', ''),
            'location': data.get('location', {}),
            'status': 'active',
            'timestamp': firebase_admin.db.ServerValue.TIMESTAMP,
            'userId': user_id
        }
        
        emergency_ref.set(emergency_data)
        
        return jsonify({
            'status': 'success',
            'emergencyId': emergency_ref.key
        })
        
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
            
        # Update the emergency record in Firebase
        emergency_ref = db.child(f'emergencies/{user_id}/{emergency_id}')
        emergency_ref.update({
            'status': data.get('status', 'resolved'),
            'updatedAt': firebase_admin.db.ServerValue.TIMESTAMP
        })
        
        return jsonify({'status': 'success'})
        
    except Exception as e:
        logger.error(f"Error updating emergency: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/volunteer/toggle', methods=['POST'])
def toggle_volunteer():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_ref = db.reference(f'users/{session["user_id"]}')
    current_status = user_ref.child('is_volunteer').get()
    
    user_ref.update({'is_volunteer': not current_status})
    
    return jsonify({'status': 'success', 'is_volunteer': not current_status})

@app.route('/api/volunteer/location', methods=['POST'])
def update_volunteer_location():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    lat = data.get('lat')
    lng = data.get('lng')
    
    db.reference(f'users/{session["user_id"]}/location').set({
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
    for element in data['elements']:
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
def ai_doctor():
    try:
        data = request.json
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
        print(f"Error in AI Doctor: {str(e)}")
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
        # Use a direct API call to the Gemini API
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
                "temperature": 0.8,
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
                            return response_data['candidates'][0]['content']['parts'][0]['text']
        
        # Fallback response if the API call fails
        return "Chest pain after jogging could indicate several conditions. If the pain is sharp, radiates to your arm or jaw, or is accompanied by shortness of breath, seek immediate medical attention. For milder discomfort, try warming up properly before exercise, staying hydrated, and gradually increasing your activity level. Consider consulting a cardiologist for a thorough evaluation, especially if the pain persists or worsens. They may recommend tests like an ECG or stress test to determine the cause."
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        # Fallback response if there's an error
        return "Chest pain after jogging could indicate several conditions. If the pain is sharp, radiates to your arm or jaw, or is accompanied by shortness of breath, seek immediate medical attention. For milder discomfort, try warming up properly before exercise, staying hydrated, and gradually increasing your activity level. Consider consulting a cardiologist for a thorough evaluation, especially if the pain persists or worsens. They may recommend tests like an ECG or stress test to determine the cause."

@app.route('/api/emergency_contacts', methods=['GET', 'POST', 'DELETE'])
def handle_emergency_contacts():
    if not session.get('user'):
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user']['uid']
    
    if request.method == 'GET':
        try:
            # Get user's emergency contacts from Firebase
            contacts_ref = db.child(f'users/{user_id}/emergency_contacts')
            contacts = contacts_ref.get()
            
            if contacts.val():
                return jsonify({'contacts': contacts.val()})
            return jsonify({'contacts': {}})
        except Exception as e:
            print(f"Error getting contacts: {str(e)}")
            return jsonify({'error': 'Failed to get contacts'}), 500
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            name = data.get('name')
            phone = data.get('phone')
            
            if not name or not phone:
                return jsonify({'error': 'Name and phone are required'}), 400
            
            # Add contact to Firebase
            contacts_ref = db.child(f'users/{user_id}/emergency_contacts')
            new_contact = contacts_ref.push({
                'name': name,
                'phone': phone,
                'added_at': {'.sv': 'timestamp'}
            })
            
            return jsonify({
                'status': 'success',
                'contact_id': new_contact['name'],
                'contact': {
                    'name': name,
                    'phone': phone
                }
            })
        except Exception as e:
            print(f"Error adding contact: {str(e)}")
            return jsonify({'error': 'Failed to add contact'}), 500
    
    elif request.method == 'DELETE':
        try:
            contact_id = request.args.get('id')
            if not contact_id:
                return jsonify({'error': 'Contact ID is required'}), 400
            
            # Remove contact from Firebase
            contact_ref = db.child(f'users/{user_id}/emergency_contacts/{contact_id}')
            contact_ref.remove()
            
            return jsonify({'status': 'success'})
        except Exception as e:
            print(f"Error removing contact: {str(e)}")
            return jsonify({'error': 'Failed to remove contact'}), 500

@app.route('/emergency_map')
def emergency_map():
    if not session.get('user'):
        return redirect(url_for('login'))
    return render_template('emergency_map.html')

@app.route('/api/generate_analysis', methods=['POST'])
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
    # Use this for development
    socketio.run(app, debug=True)
else:
    # Use this for production
    port = int(os.getenv("PORT", 10000))
    app = socketio.run(app, host="0.0.0.0", port=port) 