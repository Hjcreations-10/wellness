# Use a slim Python image for a smaller container
# Use a slim Python image for a smaller container
FROM python:3.11-slim

# Install system dependencies, including FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install the Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy language model
RUN python -m spacy download en_core_web_sm

# Copy the rest of your application code
COPY . .

# Cloud Run expects the app to listen on PORT (default 8080)
ENV PORT=8080

# Start the Streamlit app on the given port
CMD ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.enableCORS=false --server.enableXsrfProtection=false"]
