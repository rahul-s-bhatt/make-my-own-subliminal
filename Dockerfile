# Dockerfile for MindMorph Streamlit App with Piper TTS

# 1. Base Image - Use a standard Python image (e.g., 3.11 for better wheel compatibility)
# Using slim-bullseye for a smaller image size
FROM python:3.11-slim-bullseye

# 2. Set Working Directory
WORKDIR /app

# 3. Install System Dependencies
# First, update apt package lists
# Then, install packages explicitly listed here
# Use --no-install-recommends to keep the image smaller
# Clean up apt lists afterwards to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python Dependencies from requirements.txt
# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .
# Upgrade pip and install requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code
# Copy the rest of your application code into the working directory
# Use .dockerignore to exclude unnecessary files/folders
COPY . .
# Ensure your Piper voice models (.onnx, .json) are included in the copy
# (e.g., they are in the 'assets/voices/...' directory within your project)

# 6. Expose Streamlit Port
EXPOSE 8501

# 7. Set Healthcheck (Optional but recommended)
# Checks if Streamlit server is responding
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 8. Define Run Command
# Runs the Streamlit application using the main.py entry point
# Use --server.enableCORS=false if running behind certain proxies/load balancers
# Use --server.headless=true to prevent opening a browser on the server
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.enableCORS=false", "--server.headless=true"]
