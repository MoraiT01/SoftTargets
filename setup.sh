#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

ENV_NAME=".venv"

PYTHON_EXECUTABLE=$(which python3.9 || which python3.9m || which python3)

if [ -z "$PYTHON_EXECUTABLE" ]; then
    echo "ERROR: Could not find Python 3.9 executable."
    echo "Please ensure Python 3.9 is installed and in your PATH."
    exit 1
fi

echo "Using Python executable: $PYTHON_EXECUTABLE"

echo "1. Creating virtual environment: $ENV_NAME"
# Use the determined executable to create the venv
$PYTHON_EXECUTABLE -m venv $ENV_NAME

echo "2. Activating virtual environment..."
# Note: This is a setup script, we don't activate for the user's current shell, 
# we use the environment's python directly.

# Determine the correct Python path inside the venv
VENV_PYTHON="./$ENV_NAME/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    # Fallback for Windows or systems with different venv structure
    VENV_PYTHON="./$ENV_NAME/Scripts/python"
fi

echo "3. Installing the project and dependencies..."
# This command installs everything listed in setup.py
# '-e .' means editable mode, linking source files directly
$VENV_PYTHON -m pip install -U pip  # Upgrade pip inside venv
$VENV_PYTHON -m pip install -e .

echo "4. Downloading datasets"
# Replace this with your actual data download logic (e.g., using Wget/Curl or a Python script)
# For a Python-based download:
$VENV_PYTHON download_data.py

echo "Setup complete! To begin development, run:"
echo "source $ENV_NAME/bin/activate" # For Linux/Mac
# Or: source ".\.venv\Scripts\Activate.ps1" # For Windows PowerShell