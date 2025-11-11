# Start with a slim Python 3.11 image (as mentioned in README.md)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy *only* the requirements file first to leverage Docker's build cache
COPY requirements.txt ./

# Install the Python dependencies directly from requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Add the app directory to PYTHONPATH
# This is the key step that replaces 'pip install -e .'
# It allows Python to find and import modules from the 'src' directory
# (e.g., 'from src.data.dataset_loaders import ...')
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Copy the rest of the project source code into the container
COPY . .

# Run the data download script to fetch and extract the dataset
RUN python download_data.py

# Define the default command to run the main script
CMD ["python", "main.py"]