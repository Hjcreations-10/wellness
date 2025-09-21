# Use a slim Python image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Copy all project files
COPY . .

# Cloud Run expects app to listen on PORT (default 8080)
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run Streamlit on the correct port and address
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
