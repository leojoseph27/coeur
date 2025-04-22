import os
import gdown
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories if they don't exist
os.makedirs('ecg project', exist_ok=True)
os.makedirs('heart/models', exist_ok=True)
os.makedirs('archive/variables', exist_ok=True)

# File mappings with Google Drive file IDs
# Replace these with your actual Google Drive file IDs after uploading
MODEL_FILES = {
    'ecg project/best_model.pth': 'YOUR_FILE_ID_HERE',
    'heart/models/audio_model.h5': 'YOUR_FILE_ID_HERE',
    'archive/variables/variables.data-00000-of-00001': 'YOUR_FILE_ID_HERE',
    'archive/saved_model.pb': 'YOUR_FILE_ID_HERE'
}

def download_models():
    """Download model files from Google Drive if they don't exist locally."""
    
    logger.info("Checking and downloading model files...")
    
    for file_path, file_id in MODEL_FILES.items():
        # Skip if file already exists
        if os.path.exists(file_path):
            logger.info(f"File already exists: {file_path}")
            continue
            
        # Download file from Google Drive
        try:
            logger.info(f"Downloading {file_path} from Google Drive...")
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                file_path,
                quiet=False
            )
            logger.info(f"Successfully downloaded {file_path}")
        except Exception as e:
            logger.error(f"Failed to download {file_path}: {str(e)}")
            raise

if __name__ == "__main__":
    download_models() 