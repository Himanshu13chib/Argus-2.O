# Project Argus - Border Detection System

A world-class, enterprise-grade border security detection and monitoring solution designed for India's border regions.

## Architecture Overview

- **Edge Layer**: Real-time detection on Jetson Xavier NX devices
- **Control Plane**: Microservices for tracking, alerts, and analytics
- **Command Center**: React-based dashboard for operators
- **Evidence Store**: Immutable forensic data storage

## Quick Start

### Development Environment Setup

1. **Python Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Node.js Environment**:
   ```bash
   cd dashboard
   npm install
   ```

3. **Docker Development**:
   ```bash
   docker-compose up -d
   ```

4. **Verify Setup**:
   ```bash
   python verify_setup.py
   ```

## 🛠️ Development Commands

The project includes a comprehensive Makefile for development tasks:

```bash
# Setup and Environment
make setup          # Set up development environment
make dev-up         # Start development services
make dev-down       # Stop development services
make logs           # View service logs

# Building
make build          # Build all Docker images
make build-edge     # Build edge node image
make build-services # Build backend services
make build-dashboard # Build dashboard

# Testing
make test           # Run all tests
make test-edge      # Run edge node tests
make test-services  # Run backend service tests
make test-dashboard # Run dashboard tests

# Code Quality
make lint           # Run code quality checks
make format         # Format code

# Database
make db-migrate     # Run database migrations
make db-reset       # Reset database

# Maintenance
make clean          # Clean up containers and images
make reset          # Reset development environment
```

## 🏗️ Project Structure

```
project-argus/
├── shared/                     # Shared models and interfaces
│   ├── models/                 # Core data models
│   │   ├── detection.py        # Detection and bounding box models
│   │   ├── tracking.py         # Multi-object tracking models
│   │   ├── alerts.py           # Alert and crossing event models
│   │   ├── incidents.py        # Incident management models
│   │   ├── evidence.py         # Evidence and chain of custody
│   │   ├── camera.py           # Camera configuration models
│   │   ├── virtual_line.py     # Virtual line detection models
│   │   ├── health.py           # System health monitoring
│   │   └── user.py             # User management and permissions
│   └── interfaces/             # Abstract interfaces
│       ├── detection.py        # Detection pipeline interfaces
│       ├── tracking.py         # Tracking and re-ID interfaces
│       ├── alerts.py           # Alert management interfaces
│       ├── incidents.py        # Incident workflow interfaces
│       ├── evidence.py         # Evidence management interfaces
│       ├── storage.py          # Data persistence interfaces
│       ├── health.py           # Health monitoring interfaces
│       └── security.py         # Security and access control
├── edge/                       # Edge computing nodes
│   ├── src/                    # Edge node source code
│   ├── config/                 # Configuration files
│   ├── models/                 # AI model storage
│   ├── data/                   # Local data storage
│   ├── logs/                   # Edge node logs
│   ├── main.py                 # Edge node entry point
│   ├── Dockerfile              # Production edge container
│   ├── Dockerfile.simulator    # Development simulator
│   └── requirements.txt        # Edge-specific dependencies
├── services/                   # Backend microservices
│   ├── api-gateway/            # Central API gateway
│   ├── alert-service/          # Alert generation and routing
│   ├── tracking-service/       # Multi-camera tracking
│   └── evidence-service/       # Evidence management
├── dashboard/                  # React frontend
│   ├── src/                    # Dashboard source code
│   ├── public/                 # Static assets
│   ├── package.json            # Node.js dependencies
│   ├── Dockerfile              # Dashboard container
│   └── nginx.conf              # Production web server config
├── infrastructure/             # Infrastructure and deployment
│   ├── database/               # Database schemas and migrations
│   ├── monitoring/             # Monitoring and observability
│   └── scripts/                # Deployment and maintenance scripts
├── logs/                       # Application logs
├── data/                       # Application data
├── models/                     # Shared AI models
├── docker-compose.yml          # Development environment
├── Makefile                    # Development commands
├── requirements.txt            # Python dependencies
├── setup_dev.py               # Development setup script
├── verify_setup.py            # Setup verification script
└── .env.example               # Environment configuration template
```

## 🚀 Core Features

### Real-Time Detection & Tracking
- **Multi-Modal Sensors**: Visible light, thermal, and infrared cameras
- **AI-Powered Detection**: YOLOv8/YOLOv9 optimized for edge deployment
- **Cross-Camera Tracking**: Person re-identification across multiple cameras
- **Virtual Line Detection**: Configurable boundary crossing detection

### Command Center Dashboard
- **Live Video Feeds**: Real-time camera streams with detection overlays
- **Alert Management**: Comprehensive alert handling and escalation
- **Incident Workflow**: Complete case management from detection to resolution
- **Analytics Dashboard**: Historical data analysis and trend visualization

### Evidence Management
- **Immutable Storage**: HMAC-signed evidence with integrity verification
- **Chain of Custody**: Complete audit trail for legal proceedings
- **Automated Reports**: PDF and CSV forensic report generation
- **Privacy Controls**: Automatic data purging and anonymization

### Security & Compliance
- **Zero-Trust Architecture**: End-to-end encryption and micro-segmentation
- **Role-Based Access**: Operator, auditor, and administrator roles
- **Multi-Factor Authentication**: Enhanced security for sensitive operations
- **Comprehensive Auditing**: Complete system activity logging

## Requirements

- Python 3.9+
- Node.js 18+
- Docker & Docker Compose
- NVIDIA Docker runtime (for edge deployment)

## Security

This system implements zero-trust architecture with end-to-end encryption, role-based access control, and comprehensive audit logging.