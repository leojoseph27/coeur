# Coeur - Heart Disease Analysis Application

A comprehensive application for heart disease analysis, ECG processing, and emergency medical assistance.

## Features
- Heart disease prediction using machine learning
- ECG signal analysis with autoencoder
- Real-time audio analysis for heart sounds
- Emergency medical assistance system
- AI-powered medical consultation

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/leojoseph27/coeur.git
cd coeur
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Firebase credentials:
- Obtain the Firebase admin SDK JSON file
- Place it in the project root
- Set up environment variables as needed

5. Run the application:
```bash
python app.py
```

## Model Files
This project uses Git LFS for managing large model files. Make sure you have Git LFS installed:
```bash
git lfs install
```

## Environment Variables
Create a `.env` file with the following variables:
- GOOGLE_API_KEY
- Other configuration variables as needed

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[MIT License](LICENSE) 