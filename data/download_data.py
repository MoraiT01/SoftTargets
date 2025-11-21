"""This file serves to download the dataset"""

# It should only be use after the dataset is finilized

import os
import requests
import sys
import zipfile
from pathlib import Path

# --- Configuration ---
ZENODO_RECORD_ID = "17360336"
DATA_DIR = Path("data")
API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

def get_download_url(record_id: str) -> tuple[str, str]:
    """Fetches the metadata and returns the URL and filename of the first file."""
    print(f"Fetching metadata for Zenodo record ID {record_id}...")
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Zenodo metadata: {e}")
        raise

    files = data.get("files", [])
    if not files:
        raise ValueError(f"No files found in Zenodo record {record_id}.")

    # Assuming the first file listed is the main dataset to download
    file_info = files[0]
    
    # Zenodo provides a 'download' link directly under the 'links' key of the file object
    download_url = file_info["links"]["self"]
    filename = file_info["key"]
    
    print(f"Found file: {filename}")
    return download_url, filename

def download_file(url: str, filepath: Path):
    """Downloads a file from a URL to a local path with streaming."""
    print(f"Starting download to {filepath}...")
    try:
        # Use stream=True for large files
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get total file size for potential progress bar display
        total_size = int(response.headers.get('content-length', 0))
        
        # Open the local file for binary write
        with open(filepath, 'wb') as f:
            downloaded_size = 0
            chunk_size = 8192  # 8KB chunks
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded_size += len(chunk)

                # Simple progress indicator (replace with tqdm for a better one)
                if total_size > 0:
                    percent = (downloaded_size / total_size) * 100
                    print(f"Downloading: {downloaded_size/1024/1024:.2f}MB / {total_size/1024/1024:.2f}MB ({percent:.1f}%)", end='\r')

        print(f"\nDownload complete! Saved to {filepath}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        # Clean up partial file if download failed
        if filepath.exists():
            os.remove(filepath)
        raise

def main():
    try:
        # 1. Ensure the 'data' directory exists
        DATA_DIR.mkdir(exist_ok=True)
        print(f"Ensured data directory exists at: {DATA_DIR.resolve()}")
        
        # 2. Get the download URL and filename
        download_url, filename = get_download_url(ZENODO_RECORD_ID)
        
        # 3. Define the full local path
        local_filepath = DATA_DIR / filename
        
        # Skip download if the file already exists
        if local_filepath.exists():
             print(f"File already exists at {local_filepath}. Skipping download.")
             return
             
        # 4. Download the file
        download_file(download_url, local_filepath)
        
    except Exception as e:
        print(f"\nFATAL ERROR during data download: {e}")
        sys.exit(1)

    else:
        # Define the paths
        archive_path = local_filepath # Path where you saved the downloaded file
        extraction_dir = DATA_DIR / "softtarget_dataset"  # Directory where you want to extract the contents

        # 1. Create the target directory if it doesn't exist
        os.makedirs(extraction_dir, exist_ok=True)

        # 2. Extract the archive
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                print(f"Extracting {archive_path} to {extraction_dir}...")
                zip_ref.extractall(extraction_dir)
            print("Extraction complete.")
        except zipfile.BadZipFile:
            print(f"Error: {archive_path} is not a valid ZIP file or is corrupted.")
        except FileNotFoundError:
            print(f"Error: Archive file not found at {archive_path}.")

        # Write the name of the created directory to the gitignore file in the dataset folder
        with open(DATA_DIR / '.gitignore', 'a') as f:
            f.write(f"{extraction_dir.name}\n")

        # 3. (Optional) Remove the original zip file after successful extraction
        # os.remove(archive_path)

if __name__ == "__main__":
    # Ensure 'requests' is available if running this script directly
    try:
        import requests
    except ImportError:
        print("The 'requests' library is required. Please install it with: pip install requests")
        sys.exit(1)
        
    main()