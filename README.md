# Coeur - Heart Disease Analysis Application

A comprehensive application for heart disease analysis, ECG processing, and emergency medical assistance.

## Features
- Heart disease prediction using machine learning
- ECG signal analysis with autoencoder
- Real-time audio analysis for heart sounds
- Emergency medical assistance system with OpenStreetMap integration
- AI-powered medical consultation (Google Gemini)

## Tech Stack
- **Backend**: Python 3.10 / Flask + Flask-SocketIO
- **Database**: Neon PostgreSQL (auth + application data)
- **Auth**: Local password-based (werkzeug password hashing + Flask sessions)
- **AI**: Google Gemini API (gemini-2.5-flash)
- **Maps**: Leaflet + OpenStreetMap tiles, Nominatim reverse geocoding, OSRM routing
- **ML**: TensorFlow, PyTorch, scikit-learn, librosa

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/leojoseph27/coeur.git
cd coeur
```

2. Create a virtual environment (Python 3.10 required):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (see `.env.example` or create a `.env` file):
```bash
cp .env.example .env
# Edit .env with your credentials
```

5. Set up the Neon PostgreSQL database:
- Create a Neon project at https://neon.tech
- Run the schema migration (see `db_neon.py` for table definitions)
- Tables: `users`, `emergencies`, `emergency_contacts`

6. Run the application:
```bash
python app.py
```

The app runs on `http://localhost:5000`.

## Model Files
This project uses Git LFS for managing large model files. Make sure you have Git LFS installed:
```bash
git lfs install
```

Model files expected:
- `ecg project/best_model.pth` (ECG autoencoder)
- `heart/models/audio_model.h5` (audio classification)
- `heart/models/heart_model.joblib` (heart disease prediction)
- `archive/saved_model.pb` (YAMNet)

## Environment Variables
Create a `.env` file with the following variables:

| Variable | Required | Description |
|---|---|---|
| `NEON_DATABASE_URL` | **Yes** | Neon PostgreSQL connection string |
| `GOOGLE_API_KEY` | **Yes** | Google Gemini API key (for AI doctor) |
| `FLASK_SECRET_KEY` | Optional | Flask session secret (defaults to dev key) |
| `ORS_API_KEY` | Optional | OpenRouteService key (OSRM used by default) |
| `PORT` | Optional | Server port (default 5000) |
| `DEBUG` | Optional | Enable debug mode (default False) |

## Database Schema

The app uses Neon PostgreSQL with three tables:

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT,
    password_hash TEXT,
    is_volunteer BOOLEAN DEFAULT FALSE,
    location JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE emergencies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    type TEXT DEFAULT 'Emergency',
    description TEXT,
    location JSONB,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE emergency_contacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    phone TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Default Test Account
- **Email**: `leo@gmail.com`
- **Password**: `leo123`

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[MIT License](LICENSE)
