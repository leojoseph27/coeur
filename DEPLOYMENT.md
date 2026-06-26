# Deployment Guide

## ⚠️ Important: This app CANNOT run on Vercel

Vercel is a **serverless** platform designed for Next.js/Node.js frontends and
 lightweight serverless functions. This app is a **Python Flask + TensorFlow +
 PyTorch** application with large ML models — it is fundamentally incompatible
 with Vercel's architecture:

| Constraint | Vercel Limit | This App Needs |
|---|---|---|
| Serverless function size | 50 MB | ~2 GB (TensorFlow + PyTorch + models) |
| Deployment package size | 250 MB | ~3 GB |
| Function timeout | 10-60 seconds | 120+ seconds (model loading + inference) |
| Filesystem | Read-only | Needs writable `reports/` directory |
| Persistent processes | Not supported | Needs gunicorn/Flask long-running server |
| Model files at runtime | Can't download | Models must be present at build time |

**Do not attempt to deploy this app on Vercel — it will fail.**

## ✅ Recommended Platforms

### Render (recommended — config already included)

The repo includes `render.yaml` and `Procfile` pre-configured for Render.

1. Push this repo to GitHub
2. Go to https://render.com → New → Web Service → connect your GitHub repo
3. Render will auto-detect the `render.yaml` config
4. Set environment variables in the Render dashboard:
   - `NEON_DATABASE_URL` — your Neon PostgreSQL connection string
   - `GOOGLE_API_KEY` — your Google Gemini API key
   - `FLASK_SECRET_KEY` — a random secret string
5. The build will automatically:
   - Install Python dependencies (`pip install -r requirements.txt`)
   - Download model files (`python download_models.py`)
   - Start gunicorn on the assigned port

**Free tier:** Render's free tier works for testing but sleeps after 15 min
 of inactivity. For production, use a paid plan.

### Railway

```bash
railway init
railway up
```
Set the same environment variables in the Railway dashboard.

### Fly.io

```bash
fly launch
fly deploy
```

### VPS (DigitalOcean, AWS EC2, etc.)

```bash
git clone https://github.com/leojoseph27/coeur.git
cd coeur
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python download_models.py      # fetch model files
gunicorn app:app --bind 0.0.0.0:5000 --workers 1 --threads 4 --timeout 120
```
Use nginx + systemd for production.

## Required Environment Variables

| Variable | Required | Description |
|---|---|---|
| `NEON_DATABASE_URL` | **Yes** | Neon PostgreSQL connection string |
| `GOOGLE_API_KEY` | **Yes** | Google Gemini API key (for AI doctor) |
| `FLASK_SECRET_KEY` | Optional | Flask session secret (defaults to dev key) |
| `ORS_API_KEY` | Optional | OpenRouteService key (OSRM used by default) |
| `PORT` | Optional | Server port (auto-set by Render/Railway) |

## Database Setup

Before the first deploy, create the Neon schema by running the SQL in
`README.md` (the "Database Schema" section). Tables needed:
- `users` (with `password_hash` column + unique email)
- `emergencies`
- `emergency_contacts`
- `medical_info`
- `medical_records`

## Model Files

Model files are **gitignored** (too large for Git). They are downloaded at
build time by `download_models.py` from a Google Drive folder:

- **Folder** (9 files, ~13 MB): `heart/models/` contents — heart_model.joblib,
  heart_scaler.joblib, audio_model.h5, ecg_*.joblib, ecg_*.npy, scaler_params.json
- **Individual** (~31 MB): `ecg project/best_model.pth` (PyTorch ECG autoencoder)

If the YAMNet model (`archive/saved_model.pb` + `variables/`) is missing,
the heart-sound analysis feature is disabled but the rest of the app works.

## Default Test Account

After deploying, create the test user by running this once against your Neon
database (e.g., via `psql` or the Neon SQL editor):

```sql
-- The app will create users on first registration via the /register page.
-- There is no default admin account.
```
Register a new account at `/register`, then log in at `/login`.
