FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules
COPY shared/ /app/shared/

# Copy service code
COPY services/federated-learning-service/ /app/

# Create non-root user
RUN useradd -m -u 1000 flservice && chown -R flservice:flservice /app
USER flservice

EXPOSE 8007

CMD ["python", "main.py"]