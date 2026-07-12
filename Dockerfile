FROM python:3.10-slim

# Install system dependencies required for packet capture and ML libraries
RUN apt-get update && apt-get install -y \
    libpcap-dev \
    gcc \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
# Add required packages for the new architecture if not in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir nfstream streamlit plotly sqlalchemy psutil

# Copy project files
COPY . .

# Set PYTHONPATH so absolute imports work
ENV PYTHONPATH=/app

# Create data directory for SQLite
RUN mkdir -p data

# Expose API and Dashboard ports
EXPOSE 8000
EXPOSE 8501
