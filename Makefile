# Project Argus - Development Makefile

.PHONY: help setup build test clean deploy dev-up dev-down logs

# Default target
help:
	@echo "Project Argus - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  setup          - Set up development environment"
	@echo "  dev-up         - Start development environment"
	@echo "  dev-down       - Stop development environment"
	@echo "  logs           - View logs from all services"
	@echo ""
	@echo "Building:"
	@echo "  build          - Build all Docker images"
	@echo "  build-edge     - Build edge node image"
	@echo "  build-services - Build backend services"
	@echo "  build-dashboard - Build dashboard"
	@echo ""
	@echo "Testing:"
	@echo "  test           - Run all tests"
	@echo "  test-edge      - Run edge node tests"
	@echo "  test-services  - Run backend service tests"
	@echo "  test-dashboard - Run dashboard tests"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-dev     - Deploy to development environment"
	@echo "  deploy-prod    - Deploy to production environment"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean          - Clean up containers and images"
	@echo "  reset          - Reset development environment"

# Development Environment Setup
setup:
	@echo "Setting up Project Argus development environment..."
	cp .env.example .env
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  # Linux/Mac"
	@echo "  venv\\Scripts\\activate     # Windows"
	@echo ""
	@echo "Then install dependencies:"
	@echo "  pip install -r requirements.txt"
	@echo ""
	@echo "For dashboard development:"
	@echo "  cd dashboard && npm install"

# Development Environment
dev-up:
	@echo "Starting Project Argus development environment..."
	docker-compose up -d
	@echo "Services starting up..."
	@echo "Dashboard: http://localhost:3000"
	@echo "API Gateway: http://localhost:8000"
	@echo "MinIO Console: http://localhost:9001"

dev-down:
	@echo "Stopping Project Argus development environment..."
	docker-compose down

logs:
	docker-compose logs -f

# Building
build:
	@echo "Building all Project Argus images..."
	docker-compose build

build-edge:
	@echo "Building edge node image..."
	docker build -f edge/Dockerfile -t project-argus/edge:latest .

build-services:
	@echo "Building backend services..."
	docker build -f services/api-gateway/Dockerfile -t project-argus/api-gateway:latest .
	docker build -f services/alert-service/Dockerfile -t project-argus/alert-service:latest .
	docker build -f services/tracking-service/Dockerfile -t project-argus/tracking-service:latest .
	docker build -f services/evidence-service/Dockerfile -t project-argus/evidence-service:latest .

build-dashboard:
	@echo "Building dashboard..."
	docker build -f dashboard/Dockerfile -t project-argus/dashboard:latest .

# Testing
test:
	@echo "Running all tests..."
	$(MAKE) test-edge
	$(MAKE) test-services
	$(MAKE) test-dashboard

test-edge:
	@echo "Running edge node tests..."
	cd edge && python -m pytest tests/ -v

test-services:
	@echo "Running backend service tests..."
	cd services && python -m pytest tests/ -v

test-dashboard:
	@echo "Running dashboard tests..."
	cd dashboard && npm test -- --coverage --watchAll=false

# Code Quality
lint:
	@echo "Running code quality checks..."
	black --check .
	flake8 .
	mypy shared/ services/ edge/
	cd dashboard && npm run lint

format:
	@echo "Formatting code..."
	black .
	isort .
	cd dashboard && npm run format

# Database Management
db-migrate:
	@echo "Running database migrations..."
	docker-compose exec api-gateway alembic upgrade head

db-reset:
	@echo "Resetting database..."
	docker-compose exec postgres psql -U argus_user -d project_argus -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	$(MAKE) db-migrate

# Deployment
deploy-dev:
	@echo "Deploying to development environment..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

deploy-prod:
	@echo "Deploying to production environment..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Maintenance
clean:
	@echo "Cleaning up Docker containers and images..."
	docker-compose down -v
	docker system prune -f
	docker volume prune -f

reset: clean
	@echo "Resetting development environment..."
	docker-compose down -v --remove-orphans
	docker system prune -af
	$(MAKE) dev-up

# Monitoring
monitor:
	@echo "Starting monitoring stack..."
	docker-compose -f docker-compose.monitoring.yml up -d

# Backup
backup:
	@echo "Creating backup..."
	./scripts/backup.sh

# Security Scan
security-scan:
	@echo "Running security scan..."
	docker run --rm -v $(PWD):/app securecodewarrior/docker-security-scan /app

# Performance Test
perf-test:
	@echo "Running performance tests..."
	./scripts/performance-test.sh

# Documentation
docs:
	@echo "Building documentation..."
	mkdocs build
	@echo "Documentation available at site/index.html"

docs-serve:
	@echo "Serving documentation..."
	mkdocs serve