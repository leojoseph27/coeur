import os
from download_models import download_models

def test_downloads():
    """Test downloading models from Google Drive and verify they exist."""
    # Run the download function
    download_models()
    
    # Check if files were downloaded successfully
    expected_files = [
        'ecg project/best_model.pth',
        'heart/models/audio_model.h5',
        'archive/saved_model.pb',
        'archive/variables/variables.data-00000-of-00001'
    ]
    
    all_downloaded = True
    missing_files = []
    
    for file_path in expected_files:
        if not os.path.exists(file_path):
            all_downloaded = False
            missing_files.append(file_path)
    
    if all_downloaded:
        print("✅ SUCCESS: All model files downloaded successfully!")
    else:
        print(f"❌ ERROR: The following files failed to download:")
        for file in missing_files:
            print(f"  - {file}")

if __name__ == "__main__":
    test_downloads() 