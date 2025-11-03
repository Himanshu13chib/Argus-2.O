# Multi-stage build for Evidence Service
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY services/evidence-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared models and interfaces
COPY shared/ ./shared/

# Copy service code
COPY services/evidence-service/ ./

# Create directories
RUN mkdir -p /app/logs /app/evidence

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Expose port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Run the evidence service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]

# Development stage
FROM base as development

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    black \
    flake8 \
    mypy

# Override command for development with auto-reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005", "--reload"]

# Production stage
FROM base as production

# Use non-root user for security
RUN useradd -m -u 1000 argus && chown -R argus:argus /app
USER argus

# Production optimizations
ENV WORKERS=2
CMD ["gunicorn", "main:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8005"]