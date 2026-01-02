# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app ./app
COPY .env .env

# Create necessary directories
RUN mkdir -p /app/data/pdfs /app/data/vectorstore

# Expose port
EXPOSE 8800

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8800"]