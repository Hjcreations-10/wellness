# Use a slim Python image for a smaller container
FROM python:3.11-slim

# Install system dependencies, including FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy language model (optional: can be moved to requirements.txt)
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code
COPY . .

# Cloud Run expects the app to listen on $PORT
ENV PORT=8080

# Streamlit entrypoint
CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

