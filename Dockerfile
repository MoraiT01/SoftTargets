# Start with a slim Python 3.11 image (as mentioned in README.md)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy *only* the requirements file first to leverage Docker's build cache
COPY requirements.txt ./

# Install the Python dependencies directly from requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project source code into the container
COPY . .

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# Set the entrypoint to run our script when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]