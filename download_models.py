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
MODEL_FILES = {
    'ecg project/best_model.pth': '1aV_W0eBLf53-t7MseAscvwh_zLAhdwep',
    'heart/models/audio_model.h5': '1hXPE11br4jGlHK_QGWnGkD1HWKZQ_ttx',
    'archive/saved_model.pb': '1QdREwyTrDhYh7w5NkS7BO_yNYvwBFupV',
    'archive/variables/variables.data-00000-of-00001': '1Amcgy8C2-X3At6ww4IKzZHWp9r18tK9t'
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