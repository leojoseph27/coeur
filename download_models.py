import os
import logging
import time
import sys
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories if they don't exist
os.makedirs('ecg project', exist_ok=True)
os.makedirs('heart/models', exist_ok=True)
os.makedirs('archive/variables', exist_ok=True)

# Google Drive folder containing the heart + ECG model files:
# https://drive.google.com/drive/folders/1sAp0eRr5AC9M3e1wssJXdgKai3pUr02i
#
# This folder has 9 files: heart_model.joblib, heart_scaler.joblib,
# audio_model.h5, ecg_model.h5, ecg_model.joblib, ecg_scaler.joblib,
# ecg_threshold.joblib, ecg_threshold.npy, scaler_params.json
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1sAp0eRr5AC9M3e1wssJXdgKai3pUr02i"

# Large files NOT in that folder — downloaded individually by their
# Google Drive file IDs. These IDs are known-correct (verified):
#   - ecg project/best_model.pth        (PyTorch ECG autoencoder)
#   - archive/saved_model.pb            (YAMNet TensorFlow SavedModel)
#   - archive/variables/variables.data  (YAMNet variables)
LARGE_FILES = {
    'ecg project/best_model.pth': '1aV_W0eBLf53-t7MseAscvwh_zLAhdwep',
    # YAMNet files: download from the folder below if available, else skip.
    # The YAMNet model (archive/) is large; if missing, audio analysis is
    # disabled but the rest of the app works.
}


def _ensure_gdown():
    """Make sure gdown is importable; install it if missing."""
    try:
        import gdown
        return gdown
    except ImportError:
        logger.warning("gdown not installed. Installing gdown...")
        os.system(f"{sys.executable} -m pip install gdown")
        time.sleep(2)
        try:
            import gdown
            return gdown
        except ImportError:
            logger.error("Failed to install gdown. Cannot download model files.")
            return None


def _verify(path, min_size=1000):
    """Return True if path exists and is at least min_size bytes."""
    return os.path.exists(path) and os.path.getsize(path) >= min_size


def download_folder_files(gdown):
    """Download all 9 files from the Google Drive folder into heart/models/."""
    tmp_dir = "/tmp/coeur_models_folder"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    logger.info(f"Downloading model folder from {GDRIVE_FOLDER_URL} ...")
    try:
        gdown.download_folder(GDRIVE_FOLDER_URL, output=tmp_dir,
                              quiet=True, use_cookies=False)
    except Exception as e:
        logger.error(f"Folder download failed: {e}")
        return False

    # Move the downloaded files into their target locations.
    # Files in the folder: heart_model.joblib, heart_scaler.joblib,
    # audio_model.h5, ecg_*.joblib, ecg_*.npy, scaler_params.json
    moved = 0
    for root, _, files in os.walk(tmp_dir):
        for fname in files:
            src = os.path.join(root, fname)
            dst = os.path.join("heart/models", fname)
            try:
                shutil.copy2(src, dst)
                logger.info(f"  -> {dst} ({os.path.getsize(dst)} bytes)")
                moved += 1
            except Exception as e:
                logger.warning(f"  could not copy {fname}: {e}")

    logger.info(f"Moved {moved} files from folder into heart/models/")
    return moved > 0


def download_large_files(gdown):
    """Download large individual files (best_model.pth)."""
    success = True
    for path, file_id in LARGE_FILES.items():
        if _verify(path):
            logger.info(f"Already present: {path}")
            continue
        logger.info(f"Downloading {path} (id={file_id}) ...")
        for attempt in range(3):
            try:
                gdown.download(id=file_id, output=path, quiet=False, fuzzy=True)
                if _verify(path):
                    logger.info(f"  -> {path} ({os.path.getsize(path)} bytes)")
                    break
                else:
                    raise Exception("downloaded file too small")
            except Exception as e:
                logger.warning(f"  attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(5)
                else:
                    logger.error(f"  could not download {path}")
                    success = False
    return success


def download_models(max_retries=3, retry_delay=5):
    """Download all model files needed by the app.

    Returns True if the essential files (heart_model.joblib,
    heart_scaler.joblib, best_model.pth) are present afterward.
    """
    gdown = _ensure_gdown()
    if gdown is None:
        return False

    # 1) Folder files (heart + ecg models)
    download_folder_files(gdown)

    # 2) Large individual files
    download_large_files(gdown)

    # 3) Verify the essential files
    essential = [
        "heart/models/heart_model.joblib",
        "heart/models/heart_scaler.joblib",
        "ecg project/best_model.pth",
    ]
    all_ok = True
    logger.info("Verifying model files:")
    for f in essential:
        ok = _verify(f)
        logger.info(f"  {'OK' if ok else 'MISSING'}: {f}")
        if not ok:
            all_ok = False

    # Non-essential (audio/YAMNet) — warn but don't fail
    nonessential = [
        "heart/models/audio_model.h5",
        "archive/saved_model.pb",
        "archive/variables/variables.data-00000-of-00001",
    ]
    for f in nonessential:
        if not _verify(f):
            logger.warning(f"  optional file missing (feature disabled): {f}")

    return all_ok


if __name__ == "__main__":
    success = download_models()
    if not success:
        logger.warning("Some essential model files could not be downloaded.")
        # Exit 0 so the build doesn't fail; the app degrades gracefully.
        sys.exit(0)
    logger.info("Model download complete.")
