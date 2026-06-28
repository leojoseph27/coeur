# Dockerfile for Hugging Face Spaces — Coeur Heart Disease Analysis
# Python 3.10 + Flask + TensorFlow + PyTorch (CPU-only)
#
# Hugging Face Spaces Docker requirements:
#   - Must listen on 0.0.0.0:7860
#   - Must run as non-root user (uid=1000)
#   - Free tier: 16GB RAM, 2 vCPU (sufficient for TF + PyTorch)

FROM python:3.10-slim

# Install system dependencies required by librosa, scipy, and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy requirements first (better Docker layer caching)
COPY --chown=user requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps the image smaller
# The --extra-index-url in requirements.txt handles torch+cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, templates, models, and config files
COPY --chown=user . .

# Create directories that the app expects at runtime
RUN mkdir -p reports demo_files/ecg demo_files/heart_sound

# Hugging Face Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860

# Disable TensorFlow GPU (we're CPU-only in Docker)
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Run with gunicorn — sync worker (eventlet/gevent crash with TF native libs)
# 1 worker, 4 threads, 300s timeout (TF import on first audio request is slow)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "300"]
