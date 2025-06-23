# Use a base image with Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies needed by pydub (ffmpeg)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy app files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable to avoid .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Set environment variable to unbuffer stdout/stderr
ENV PYTHONUNBUFFERED=1

# Run your app (adjust if you use Flask, FastAPI, etc.)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
