# Project Argus Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying Project Argus Border Detection System across different environments including development, staging, and production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Environment](#development-environment)
3. [Staging Environment](#staging-environment)
4. [Production Environment](#production-environment)
5. [Edge Node Deployment](#edge-node-deployment)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Security Configuration](#security-configuration)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

#### Command Center (Minimum)
- **CPU**: 16 cores (Intel Xeon or AMD EPYC)
- **RAM**: 64GB DDR4
- **Storage**: 2TB NVMe SSD (system) + 10TB HDD (evidence storage)
- **Network**: 10Gbps Ethernet
- **GPU**: NVIDIA RTX 4080 or better (for analytics)

#### Edge Nodes (Per Camera)
- **Primary**: NVIDIA Jetson Xavier NX (16GB)
- **Alternative**: Intel NUC with Intel NPU or Google Coral Dev Board
- **Storage**: 512GB NVMe SSD
- **Network**: Gigabit Ethernet + 4G/5G backup
- **Power**: UPS with 4-hour backup capacity

#### Network Infrastructure
- **Bandwidth**: Minimum 50Mbps per camera feed
- **Latency**: <100ms between edge nodes and command center
- **Redundancy**: Dual network paths with automatic failover

### Software Prerequisites

#### Operating System Support
- **Linux**: Ubuntu 20.04 LTS or 22.04 LTS (recommended)
- **Container Runtime**: Docker 24.0+ with NVIDIA Container Toolkit
- **Orchestration**: Kubernetes 1.28+ or Docker Compose 2.20+

#### Dependencies
```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Install Kubernetes (optional, for production)
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update && sudo apt-get install -y kubelet kubeadm kubectl
```

## Development Environment

### Quick Start

1. **Clone Repository**
```bash
git clone https://github.com/your-org/project-argus.git
cd project-argus
```

2. **Environment Configuration**
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

3. **Start Development Stack**
```bash
# Start all services
make dev-up

# Or using Docker Compose directly
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

4. **Verify Installation**
```bash
# Run setup verification
python verify_setup.py

# Check service health
curl http://localhost:8000/health
```

### Development Services

The development environment includes:
- **API Gateway**: http://localhost:8000
- **Dashboard**: http://localhost:3000
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
- **MinIO**: http://localhost:9000
- **Grafana**: http://localhost:3001

### Development Workflow

```bash
# Start development environment
make dev-up

# View logs
make logs

# Run tests
make test

# Stop environment
make dev-down

# Clean up (removes volumes)
make clean
```

## Staging Environment

### Infrastructure Setup

1. **Server Preparation**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y curl wget git htop iotop

# Configure firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8000/tcp
sudo ufw --force enable
```

2. **SSL Certificate Setup**
```bash
# Generate certificates
cd infrastructure/tls
./generate-certs.sh staging.projectargus.com

# Or use Let's Encrypt
sudo apt install certbot
sudo certbot certonly --standalone -d staging.projectargus.com
```

3. **Deploy Staging Stack**
```bash
# Set environment
export ENVIRONMENT=staging

# Deploy services
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d

# Initialize database
docker-compose exec database psql -U argus -d argus -f /docker-entrypoint-initdb.d/01-init.sql
```

### Staging Configuration

Create `docker-compose.staging.yml`:
```yaml
version: '3.8'

services:
  api-gateway:
    environment:
      - ENVIRONMENT=staging
      - LOG_LEVEL=INFO
      - DATABASE_URL=postgresql://argus:${DB_PASSWORD}@database:5432/argus
    
  dashboard:
    environment:
      - REACT_APP_API_URL=https://staging-api.projectargus.com
      - REACT_APP_ENVIRONMENT=staging

  database:
    environment:
      - POSTGRES_DB=argus_staging
    volumes:
      - staging_db_data:/var/lib/postgresql/data

volumes:
  staging_db_data:
```

## Production Environment

### High Availability Setup

#### Load Balancer Configuration (NGINX)

```nginx
# /etc/nginx/sites-available/projectargus
upstream api_backend {
    server api-gateway-1:8000 weight=3;
    server api-gateway-2:8000 weight=3;
    server api-gateway-3:8000 weight=2 backup;
}

upstream dashboard_backend {
    server dashboard-1:3000;
    server dashboard-2:3000;
}

server {
    listen 443 ssl http2;
    server_name api.projectargus.com;
    
    ssl_certificate /etc/ssl/certs/projectargus.crt;
    ssl_certificate_key /etc/ssl/private/projectargus.key;
    ssl_protocols TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}

server {
    listen 443 ssl http2;
    server_name dashboard.projectargus.com;
    
    ssl_certificate /etc/ssl/certs/projectargus.crt;
    ssl_certificate_key /etc/ssl/private/projectargus.key;
    
    location / {
        proxy_pass http://dashboard_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Kubernetes Deployment

1. **Cluster Setup**
```bash
# Initialize cluster
sudo kubeadm init --pod-network-cidr=10.244.0.0/16

# Configure kubectl
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Install network plugin (Flannel)
kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
```

2. **Deploy Application**
```bash
# Apply configurations
kubectl apply -f infrastructure/k8s/

# Verify deployment
kubectl get pods -n argus
kubectl get services -n argus
```

#### Database Clustering (PostgreSQL)

```yaml
# postgresql-cluster.yml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: postgres-cluster
  namespace: argus
spec:
  instances: 3
  
  postgresql:
    parameters:
      max_connections: "200"
      shared_buffers: "256MB"
      effective_cache_size: "1GB"
      
  bootstrap:
    initdb:
      database: argus
      owner: argus
      secret:
        name: postgres-credentials
        
  storage:
    size: 1Ti
    storageClass: fast-ssd
    
  monitoring:
    enabled: true
```

### Production Deployment Steps

1. **Infrastructure Provisioning**
```bash
# Create namespace
kubectl create namespace argus

# Create secrets
kubectl create secret generic postgres-credentials \
  --from-literal=username=argus \
  --from-literal=password=${POSTGRES_PASSWORD} \
  -n argus

kubectl create secret generic jwt-secret \
  --from-literal=secret=${JWT_SECRET} \
  -n argus
```

2. **Deploy Core Services**
```bash
# Deploy database
kubectl apply -f infrastructure/k8s/database.yaml

# Deploy Redis
kubectl apply -f infrastructure/k8s/redis.yaml

# Deploy MinIO
kubectl apply -f infrastructure/k8s/minio.yaml

# Wait for storage to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n argus --timeout=300s
```

3. **Deploy Application Services**
```bash
# Deploy microservices
kubectl apply -f infrastructure/k8s/services.yaml

# Deploy API Gateway
kubectl apply -f infrastructure/k8s/api-gateway.yaml

# Deploy Dashboard
kubectl apply -f infrastructure/k8s/dashboard.yaml

# Deploy ingress
kubectl apply -f infrastructure/k8s/ingress.yaml
```

4. **Verify Deployment**
```bash
# Check all pods are running
kubectl get pods -n argus

# Check services
kubectl get services -n argus

# Check ingress
kubectl get ingress -n argus

# Test health endpoint
curl https://api.projectargus.com/health
```

## Edge Node Deployment

### NVIDIA Jetson Xavier NX Setup

1. **Flash JetPack**
```bash
# Download NVIDIA SDK Manager
# Flash JetPack 5.1.2 with all components

# Verify installation
sudo jetson_release
nvidia-smi
```

2. **Install Dependencies**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Runtime
sudo apt install nvidia-container-runtime
```

3. **Deploy Edge Services**
```bash
# Copy edge configuration
scp -r edge/ jetson@edge-node-01:~/

# SSH to edge node
ssh jetson@edge-node-01

# Build and start edge services
cd edge/
docker build -t argus-edge .
docker run -d --name argus-edge \
  --runtime nvidia \
  --device /dev/video0 \
  --device /dev/video1 \
  -v /data:/data \
  -e COMMAND_CENTER_URL=https://api.projectargus.com \
  -e CAMERA_CONFIG=/data/camera-config.json \
  argus-edge
```

### Edge Configuration

Create `/data/camera-config.json`:
```json
{
  "cameras": [
    {
      "id": "cam-001",
      "name": "Border Sector Alpha",
      "type": "visible",
      "device": "/dev/video0",
      "resolution": "1920x1080",
      "fps": 30,
      "location": {
        "latitude": 28.6139,
        "longitude": 77.2090
      }
    },
    {
      "id": "cam-002", 
      "name": "Border Sector Alpha Thermal",
      "type": "thermal",
      "device": "/dev/video1",
      "resolution": "640x480",
      "fps": 30,
      "location": {
        "latitude": 28.6139,
        "longitude": 77.2090
      }
    }
  ],
  "virtual_lines": [
    {
      "id": "line-001",
      "camera_id": "cam-001",
      "points": [
        {"x": 100, "y": 300},
        {"x": 1820, "y": 300}
      ],
      "direction": "bidirectional",
      "sensitivity": 0.8
    }
  ],
  "detection": {
    "model_path": "/models/yolov8n.pt",
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4
  },
  "tracking": {
    "max_age": 30,
    "min_hits": 3,
    "iou_threshold": 0.3
  }
}
```

### Edge Monitoring

```bash
# Monitor edge node health
docker logs argus-edge --tail 100 -f

# Check GPU utilization
nvidia-smi

# Monitor system resources
htop
iotop

# Check network connectivity
ping api.projectargus.com
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerting-rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'argus-api'
    static_configs:
      - targets: ['api-gateway:8000']
    metrics_path: /metrics
    
  - job_name: 'argus-edge'
    static_configs:
      - targets: ['edge-node-01:8080', 'edge-node-02:8080']
    metrics_path: /metrics
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### Grafana Dashboards

Deploy monitoring stack:
```bash
# Deploy monitoring
kubectl apply -f infrastructure/monitoring/

# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n monitoring

# Import dashboards
# - Argus Overview: infrastructure/monitoring/grafana-dashboards/argus-overview.json
# - Detection Performance: infrastructure/monitoring/grafana-dashboards/detection-performance.json
```

### Log Aggregation

```yaml
# loki-config.yml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
```

## Security Configuration

### TLS Certificate Management

```bash
# Generate CA certificate
openssl genrsa -out ca-key.pem 4096
openssl req -new -x509 -days 365 -key ca-key.pem -sha256 -out ca.pem

# Generate server certificate
openssl genrsa -out server-key.pem 4096
openssl req -subj "/CN=api.projectargus.com" -sha256 -new -key server-key.pem -out server.csr
openssl x509 -req -days 365 -sha256 -in server.csr -CA ca.pem -CAkey ca-key.pem -out server-cert.pem
```

### Network Security

```bash
# Configure firewall rules
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw allow 443/tcp
sudo ufw allow from 10.0.0.0/8 to any port 5432  # Database access
sudo ufw --force enable

# Configure fail2ban
sudo apt install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### Secrets Management

```bash
# Create Kubernetes secrets
kubectl create secret tls argus-tls \
  --cert=server-cert.pem \
  --key=server-key.pem \
  -n argus

kubectl create secret generic argus-secrets \
  --from-literal=database-password=${DB_PASSWORD} \
  --from-literal=jwt-secret=${JWT_SECRET} \
  --from-literal=encryption-key=${ENCRYPTION_KEY} \
  -n argus
```

## Troubleshooting

### Common Issues

#### 1. Edge Node Connection Issues
```bash
# Check network connectivity
ping api.projectargus.com
telnet api.projectargus.com 443

# Check DNS resolution
nslookup api.projectargus.com

# Check certificates
openssl s_client -connect api.projectargus.com:443 -servername api.projectargus.com
```

#### 2. High Memory Usage
```bash
# Check memory usage
free -h
docker stats

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full ./your-app

# Restart services if needed
docker-compose restart
```

#### 3. Database Connection Issues
```bash
# Check database status
kubectl get pods -l app=postgres -n argus
kubectl logs postgres-0 -n argus

# Test connection
psql -h localhost -U argus -d argus -c "SELECT version();"

# Check connection pool
kubectl exec -it postgres-0 -n argus -- psql -U argus -c "SELECT * FROM pg_stat_activity;"
```

#### 4. Performance Issues
```bash
# Check system resources
top
iotop
nvidia-smi

# Check application metrics
curl http://localhost:8000/metrics

# Analyze logs
kubectl logs -f deployment/api-gateway -n argus
```

### Log Analysis

```bash
# View application logs
kubectl logs -f deployment/api-gateway -n argus --tail=100

# Search for errors
kubectl logs deployment/api-gateway -n argus | grep ERROR

# Export logs for analysis
kubectl logs deployment/api-gateway -n argus --since=1h > api-gateway.log
```

### Performance Tuning

#### Database Optimization
```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM detections WHERE camera_id = 'cam-001' AND timestamp > NOW() - INTERVAL '1 hour';

-- Create indexes
CREATE INDEX CONCURRENTLY idx_detections_camera_timestamp ON detections(camera_id, timestamp);
CREATE INDEX CONCURRENTLY idx_alerts_timestamp ON alerts(timestamp);

-- Update statistics
ANALYZE;
```

#### Application Tuning
```yaml
# Increase resource limits
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"

# Configure JVM for Java services
env:
  - name: JAVA_OPTS
    value: "-Xmx2g -Xms1g -XX:+UseG1GC"
```

## Backup and Recovery

### Database Backup
```bash
# Create backup
kubectl exec postgres-0 -n argus -- pg_dump -U argus argus > backup-$(date +%Y%m%d).sql

# Restore backup
kubectl exec -i postgres-0 -n argus -- psql -U argus argus < backup-20231201.sql
```

### Evidence Backup
```bash
# Sync evidence to backup location
rsync -av --progress /data/evidence/ backup-server:/backups/evidence/

# Verify backup integrity
find /backups/evidence -name "*.mp4" -exec ffprobe {} \; 2>&1 | grep -i error
```

This deployment guide provides comprehensive instructions for setting up Project Argus across different environments while maintaining security, scalability, and reliability standards.