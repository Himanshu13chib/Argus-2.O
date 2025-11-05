# 🚀 Project Argus 2.0 - Installation Guide

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10/11
- **CPU**: 8+ cores (Intel i7/AMD Ryzen 7 or better)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 100GB+ SSD storage
- **GPU**: NVIDIA GTX 1660+ (optional, for AI acceleration)

### Software Dependencies
- **Python**: 3.8+ with pip
- **Node.js**: 16+ with npm
- **Docker**: 20.10+ with Docker Compose
- **Kubernetes**: 1.24+ (for production deployment)
- **Git**: Latest version

## 🎯 Quick Start (Development)

### 1. Clone Repository
```bash
git clone https://github.com/Himanshu13chib/Argus-2.O.git
cd Argus-2.O
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Install Python dependencies
pip install -r requirements.txt

# Install dashboard dependencies
cd dashboard
npm install
cd ..
```

### 3. Start Application
```bash
# Option 1: Full application (recommended)
python start_real_app.py

# Option 2: Individual services
python run_full_app.py
```

### 4. Access Application
- **Dashboard**: http://localhost:3000
- **API Gateway**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🐳 Docker Deployment

### Development Environment
```bash
# Build and start all services
docker-compose up --build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f
```

### Production Environment
```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale api-gateway=3
```

## ☸️ Kubernetes Deployment

### Prerequisites
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### Deploy to Kubernetes
```bash
# Create namespace
kubectl create namespace argus

# Deploy with Helm
helm install argus-2 ./infrastructure/helm/argus-2.0 -n argus

# Check deployment status
kubectl get pods -n argus
kubectl get services -n argus
```

### Access Services
```bash
# Port forward for local access
kubectl port-forward -n argus svc/api-gateway 8000:8000
kubectl port-forward -n argus svc/dashboard 3000:3000

# Or use ingress (if configured)
kubectl get ingress -n argus
```

## ☁️ Cloud Deployment

### AWS Deployment
```bash
# Configure AWS CLI
aws configure

# Deploy using EKS
./infrastructure/scripts/deploy-aws.sh

# Access via Load Balancer
aws elbv2 describe-load-balancers --region us-east-1
```

### Azure Deployment
```bash
# Login to Azure
az login

# Deploy using AKS
./infrastructure/scripts/deploy-azure.sh

# Get service endpoints
az aks get-credentials --resource-group argus-rg --name argus-cluster
kubectl get services
```

### Google Cloud Deployment
```bash
# Authenticate with GCP
gcloud auth login

# Deploy using GKE
./infrastructure/scripts/deploy-gcp.sh

# Access services
gcloud container clusters get-credentials argus-cluster --zone us-central1-a
```

## 🔧 Configuration

### Environment Variables
```bash
# Core Application Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
API_VERSION=v2

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/argus
REDIS_URL=redis://localhost:6379/0

# Security Settings
SECRET_KEY=your-super-secret-key-change-in-production
JWT_SECRET_KEY=jwt-secret-key-change-in-production
ENCRYPTION_KEY=your-32-byte-encryption-key-here

# AI/ML Configuration
MODEL_PATH=./models
CONFIDENCE_THRESHOLD=0.8
NMS_THRESHOLD=0.4

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
JAEGER_ENABLED=true
```

### Database Setup
```bash
# PostgreSQL (Production)
sudo apt-get install postgresql postgresql-contrib
sudo -u postgres createdb argus
sudo -u postgres createuser argus_user

# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser
```

### Redis Setup
```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test connection
redis-cli ping
```

## 🔐 Security Configuration

### SSL/TLS Setup
```bash
# Generate certificates
./infrastructure/tls/generate-certs.sh

# Configure nginx
sudo cp infrastructure/nginx/nginx.conf /etc/nginx/sites-available/argus
sudo ln -s /etc/nginx/sites-available/argus /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### Firewall Configuration
```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8000/tcp  # API Gateway
sudo ufw enable

# iptables
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4
```

## 📊 Monitoring Setup

### Prometheus & Grafana
```bash
# Deploy monitoring stack
kubectl apply -f infrastructure/monitoring/

# Access Grafana
kubectl port-forward -n monitoring svc/grafana 3001:3000

# Default credentials: admin/admin
```

### Log Aggregation
```bash
# Deploy ELK stack
docker-compose -f infrastructure/monitoring/docker-compose.monitoring.yml up -d

# Access Kibana
open http://localhost:5601
```

## 🧪 Testing

### Run Tests
```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python tests/integration/run_integration_tests.py

# Performance tests
python tests/performance/run_load_tests.py

# Security tests
python tests/security/run_security_tests.py
```

### Load Testing
```bash
# Install k6
sudo apt-get install k6

# Run load tests
k6 run tests/performance/api_load_test.js
```

## 🔄 Migration from v1.0

### Backup Current System
```bash
# Backup database
./infrastructure/scripts/backup.sh

# Export configuration
python scripts/export_config.py > config_backup.json
```

### Migration Process
```bash
# Run migration script
python scripts/migrate_v1_to_v2.py

# Verify migration
python scripts/verify_migration.py

# Update configuration
cp .env.v1 .env.v2.backup
cp .env.example .env
# Update .env with your values
```

## 🚨 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>
```

#### Docker Issues
```bash
# Clean Docker system
docker system prune -a

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up
```

#### Database Connection Issues
```bash
# Check database status
sudo systemctl status postgresql

# Reset database
dropdb argus && createdb argus
python manage.py migrate
```

#### Memory Issues
```bash
# Check memory usage
free -h

# Increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Performance Optimization

#### System Tuning
```bash
# Increase file limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize network
echo "net.core.rmem_max = 134217728" >> /etc/sysctl.conf
echo "net.core.wmem_max = 134217728" >> /etc/sysctl.conf
sysctl -p
```

#### Database Optimization
```bash
# PostgreSQL tuning
echo "shared_buffers = 256MB" >> /etc/postgresql/13/main/postgresql.conf
echo "effective_cache_size = 1GB" >> /etc/postgresql/13/main/postgresql.conf
sudo systemctl restart postgresql
```

## 📞 Support

### Getting Help
- **Documentation**: [docs/](docs/)
- **GitHub Issues**: [Issues](https://github.com/Himanshu13chib/Argus-2.O/issues)
- **Community**: [Discussions](https://github.com/Himanshu13chib/Argus-2.O/discussions)
- **Email**: support@projectargus.com

### Professional Support
- **Enterprise Support**: Available for production deployments
- **Training**: On-site and remote training available
- **Consulting**: Custom implementation and integration services

---

**Project Argus 2.0** - Advanced AI-powered border security system