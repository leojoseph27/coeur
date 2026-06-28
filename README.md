# Coeur - Heart Disease Analysis Application

A comprehensive application for heart disease analysis, ECG processing, and emergency medical assistance.

## Features
- Heart disease prediction using machine learning (RandomForest, 500 trees)
- ECG signal analysis with PyTorch autoencoder
- Real-time audio analysis for heart sounds (TensorFlow + YAMNet)
- Emergency medical assistance system with OpenStreetMap integration
- AI-powered medical consultation (Google Gemini 2.5 Flash)
- Professional cardiovascular analysis report with evidence tables
- Demo dataset manager for demonstrations
- Neon PostgreSQL database (auth + application data)

## Tech Stack
- **Backend**: Python 3.10 / Flask + Flask-SocketIO
- **Database**: Neon PostgreSQL
- **AI**: Google Gemini API (google-genai SDK, gemini-2.5-flash)
- **ML**: TensorFlow 2.12, PyTorch 2.0.1+cpu, scikit-learn, librosa
- **Maps**: Leaflet + OpenStreetMap, Nominatim, OSRM
- **Deployment**: Hugging Face Spaces (Docker) or Render

## Hugging Face Spaces Deployment

### 1. Create a new Space
- Go to https://huggingface.co/new-space
- Name: `coeur` (or any name)
- License: MIT
- SDK: **Docker**
- Hardware: **CPU basic (16GB RAM)** — Free tier
- Visibility: Public or Private

### 2. Clone the Space and push code
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/coeur
cd coeur
# Copy all files from this repo into the Space repo
git add .
git commit -m "Deploy to Hugging Face Spaces"
git push
```

### 3. Set Environment Variables (Secrets)
In the Hugging Face Space settings → **Variables and secrets**:

| Secret Name | Value |
|---|---|
| `NEON_DATABASE_URL` | `postgresql://neondb_owner:...@ep-xxx.neon.tech/neondb?sslmode=require` |
| `GOOGLE_API_KEY` | Your Google Gemini API key (from https://aistudio.google.com/app/apikey) |
| `FLASK_SECRET_KEY` | A random 64-char hex string (`python -c "import secrets; print(secrets.token_hex(32))"`) |

### 4. The Space will auto-build
- Dockerfile installs Python 3.10 + all dependencies
- Gunicorn starts on port 7860
- App is live at `https://YOUR_USERNAME-coeur.hf.space`

### Default Test Account
- Email: `leo@gmail.com`
- Password: `leo123`

## Render Deployment (alternative)
See `DEPLOYMENT.md` for Render deployment instructions.

## Local Development
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Fill in your credentials
python app.py
```
App runs on `http://localhost:7860`.

## Database Schema
Neon PostgreSQL with 7 tables:
- `users` (id, email, name, password_hash, is_admin, is_volunteer, location)
- `emergencies` (id, user_id, type, description, location, status)
- `emergency_contacts` (id, user_id, name, phone)
- `medical_info` (id, user_id, medications, allergies)
- `medical_records` (id, user_id, name, type, content)
- `demo_ecg_files` (id, title, filename, storage_path, description)
- `demo_heart_sound_files` (id, title, filename, storage_path, description)

## Model Files
All model files are committed to the repository (total ~60MB):
- `heart/models/heart_model.joblib` — RandomForest heart disease model
- `heart/models/heart_scaler.joblib` — MinMaxScaler
- `heart/models/audio_model.h5` — Keras heart sound classifier
- `ecg project/best_model.pth` — PyTorch ECG autoencoder
- `archive/saved_model.pb` + `archive/variables/` — YAMNet embeddings model

## Architecture
- All ML libraries are **lazy-loaded** (imported inside functions, not at startup)
- Startup RAM: ~80MB (Flask only)
- Peak RAM: ~950MB (when TensorFlow is loaded for heart sound analysis)
- Hugging Face Spaces Free tier provides 16GB RAM — sufficient for all features

## License
MIT
