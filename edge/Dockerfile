# Multi-stage build for Edge Node deployment
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY edge/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional ML libraries for Jetson
RUN pip3 install --no-cache-dir \
    ultralytics \
    onnxruntime-gpu \
    tensorrt \
    pycuda

# Copy shared models and interfaces
COPY shared/ ./shared/

# Copy edge application code
COPY edge/ ./

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV TRT_LOGGER_LEVEL=ERROR

# Expose ports
EXPOSE 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the edge detection service
CMD ["python3", "main.py"]

# Development stage with additional tools
FROM base as development

# Install development tools
RUN pip3 install --no-cache-dir \
    pytest \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    jupyter

# Copy test data and scripts
COPY edge/tests/ ./tests/
COPY edge/scripts/ ./scripts/

# Override command for development
CMD ["python3", "main.py", "--debug"]

# Production stage - optimized and minimal
FROM base as production

# Remove unnecessary packages and clean up
RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only production code
COPY edge/src/ ./src/
COPY edge/config/ ./config/

# Set production environment
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

# Use non-root user for security
RUN useradd -m -u 1000 argus && chown -R argus:argus /app
USER argus

# Production command
CMD ["python3", "src/main.py"]