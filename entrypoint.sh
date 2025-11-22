#!/bin/bash
set -e

# 1. Run the data downloader script.
# This script is smart: it will only download the data if it's not
# already present in the mounted /app/data/cifar100_data folder.
echo "Running data check/download..."
python /app/data/download_data.py
echo "Creating corresponding CSV files..."
python /app/data/create_csv.py

# 2. Run the main training script.
# This script contains Task.init() and will connect to the ClearML
# server using the environment variables passed to 'docker run'.
echo "Starting training script..."
python /app/main.py

echo "Script finished."