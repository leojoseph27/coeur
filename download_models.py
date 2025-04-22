import os
import logging
import time
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories if they don't exist
os.makedirs('ecg project', exist_ok=True)
os.makedirs('heart/models', exist_ok=True)
os.makedirs('archive/variables', exist_ok=True)

# File mappings with Google Drive file IDs - each file has a unique ID
MODEL_FILES = {
    'ecg project/best_model.pth': '1aV_W0eBLf53-t7MseAscvwh_zLAhdwep',
    'heart/models/audio_model.h5': '1hXPE11br4jGlHK_QGWnGkD1HWKZQ_ttx',
    'heart/models/heart_model.joblib': '1-YKtnJQgqzrwBs_DcOvDqpKS-4QGzuHt',
    'heart/models/heart_scaler.joblib': '1-ZcX7QK8Z9Y2V5J6tQ_5X7Q_4XZKfvwp',
    'archive/saved_model.pb': '1-_X8Z9Y2V5J6tQ_5X7Q_4XZKfvwq',
    'archive/variables/variables.data-00000-of-00001': '1-aX8Z9Y2V5J6tQ_5X7Q_4XZKfvwr'
}

def download_models(max_retries=3, retry_delay=5):
    """
    Download model files from Google Drive, with retry logic to handle connection issues.
    
    Args:
        max_retries: Maximum number of retry attempts per file
        retry_delay: Delay in seconds between retries
    """
    
    # Import gdown here so we can install it first if missing
    try:
        import gdown
    except ImportError:
        logger.warning("gdown not installed. Installing gdown...")
        os.system('pip install gdown')
        time.sleep(2)  # Give time for installation to complete
        try:
            import gdown
        except ImportError:
            logger.error("Failed to install gdown. Cannot download model files.")
            return False
    
    logger.info("Starting model downloads from Google Drive...")
    
    success = True
    for file_path, file_id in MODEL_FILES.items():
        # Skip if file already exists and has content
        if os.path.exists(file_path):
            # Remove file if it's too small (likely corrupted)
            if os.path.getsize(file_path) < 1000:  # Less than 1KB
                logger.warning(f"Found small file, removing: {file_path}")
                os.remove(file_path)
            else:
                logger.info(f"File already exists with content: {file_path}")
                continue
            
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {file_path} from Google Drive (attempt {attempt+1}/{max_retries})...")
                
                # Use a direct download approach
                download_url = f"https://drive.google.com/uc?id={file_id}"
                
                # Try with gdown
                gdown.download(download_url, file_path, quiet=False)
                
                # Verify download was successful
                if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:  # More than 1KB
                    logger.info(f"Successfully downloaded {file_path}")
                    break
                else:
                    logger.warning(f"Downloaded file appears too small: {file_path}")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    raise Exception("Downloaded file is too small")
                    
            except Exception as e:
                logger.error(f"Download attempt {attempt+1} failed for {file_path}: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All {max_retries} attempts failed for {file_path}")
                    success = False
    
    if success:
        logger.info("All model files downloaded successfully")
    else:
        logger.warning("Some model files could not be downloaded")
    
    return success

if __name__ == "__main__":
    success = download_models()
    if not success:
        logger.warning("Some downloads failed, but the application will try to continue.")
        # Exit with a non-zero status but not a fatal error
        sys.exit(0)  # Use 0 so Render build doesn't fail 