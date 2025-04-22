import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Files to remove
FILES_TO_REMOVE = [
    'ecg project/best_model.pth',
    'heart/models/audio_model.h5',
    'archive/variables/variables.data-00000-of-00001',
    'archive/saved_model.pb'
]

def cleanup():
    """Remove large model files from the repository."""
    for file_path in FILES_TO_REMOVE:
        if os.path.exists(file_path):
            try:
                logger.info(f"Removing {file_path}...")
                os.remove(file_path)
                logger.info(f"Successfully removed {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {str(e)}")

if __name__ == "__main__":
    cleanup()
    print("Large files removed. Now you can commit your changes without LFS quota issues.") 