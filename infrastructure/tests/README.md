# Project Argus Deployment Tests

This directory contains comprehensive tests for Project Argus deployment infrastructure, including container builds, orchestration, scaling, monitoring, and system integration.

## Test Categories

### 1. Container Build Tests (`test_deployment.py::TestContainerBuilds`)
- Tests Docker container builds for all services
- Validates multi-stage builds and production targets
- Ensures all required dependencies are included

### 2. Docker Compose Tests (`test_deployment.py::TestDockerCompose`)
- Tests Docker Compose deployment procedures
- Validates service health checks and connectivity
- Tests database initialization and service dependencies

### 3. Kubernetes Tests (`test_deployment.py::TestKubernetesDeployment`)
- Validates Kubernetes manifest syntax and structure
- Tests Horizontal Pod Autoscaler configurations
- Ensures proper resource limits and requests

### 4. Scaling Tests (`test_deployment.py::TestScalingCapabilities`)
- Tests horizontal scaling with Docker Compose
- Validates load balancer configurations
- Tests scaling policies and thresholds

### 5. Monitoring Tests (`test_monitoring.py`)
- Tests Prometheus configuration and metrics collection
- Validates Grafana dashboard configurations
- Tests AlertManager routing and notification setup
- Validates Jaeger tracing configuration

### 6. Failover Tests (`test_deployment.py::TestFailoverCapabilities`)
- Tests database failover and recovery
- Validates service restart and recovery procedures
- Tests system resilience under failure conditions

### 7. Integration Tests (`test_integration.py`)
- Tests end-to-end deployment scenarios
- Validates service discovery and communication
- Tests system performance baselines
- Validates security configurations

## Running Tests

### Prerequisites
```bash
# Install test dependencies
pip install -r infrastructure/tests/requirements.txt

# Ensure Docker and Docker Compose are installed
docker --version
docker-compose --version

# For Kubernetes tests, ensure kubectl is available
kubectl version --client
```

### Run All Tests
```bash
./infrastructure/tests/run_tests.sh
```

### Run Specific Test Categories
```bash
# Container build tests only
./infrastructure/tests/run_tests.sh build

# Kubernetes manifest tests only
./infrastructure/tests/run_tests.sh k8s

# Monitoring system tests only
./infrastructure/tests/run_tests.sh monitoring

# Scaling capability tests only
./infrastructure/tests/run_tests.sh scaling

# Failover tests only
./infrastructure/tests/run_tests.sh failover
```

### Run Individual Test Files
```bash
# Run deployment tests
pytest infrastructure/tests/test_deployment.py -v

# Run monitoring tests
pytest infrastructure/tests/test_monitoring.py -v

# Run integration tests
pytest infrastructure/tests/test_integration.py -v
```

## Test Environment Setup

### Local Development
Tests are designed to run against local Docker containers and services. The test runner automatically:
- Creates isolated test environments
- Starts required services
- Runs tests with proper cleanup
- Removes test containers and volumes

### CI/CD Integration
Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Deployment Tests
on: [push, pull_request]
jobs:
  test-deployment:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r infrastructure/tests/requirements.txt
      - name: Run deployment tests
        run: ./infrastructure/tests/run_tests.sh
```

## Test Configuration

### Environment Variables
- `POSTGRES_PASSWORD`: Database password for testing
- `REDIS_PASSWORD`: Redis password for testing
- `GRAFANA_PASSWORD`: Grafana admin password for testing

### Test Data
Tests use isolated test databases and temporary volumes to avoid conflicts with development data.

### Timeouts
Tests include appropriate timeouts for service startup and health checks:
- Database services: 30 seconds
- Application services: 45 seconds
- Monitoring services: 60 seconds

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   - Ensure no services are running on test ports (8000-8005, 3000, 9090, etc.)
   - Use `docker-compose down` to stop any running services

2. **Resource Constraints**
   - Tests require sufficient memory and CPU for multiple containers
   - Minimum 8GB RAM recommended for full test suite

3. **Network Issues**
   - Ensure Docker daemon is running
   - Check firewall settings for container networking

4. **Permission Issues**
   - Ensure user has Docker permissions
   - Run with appropriate privileges for volume mounts

### Debug Mode
Run tests with verbose output and no cleanup:
```bash
pytest infrastructure/tests/ -v -s --tb=long
```

### Manual Cleanup
If tests fail to cleanup automatically:
```bash
# Stop all containers
docker-compose -f docker-compose.yml down -v
docker-compose -f infrastructure/monitoring/docker-compose.monitoring.yml down -v

# Remove test images
docker images -q "project-argus/*:test" | xargs -r docker rmi -f

# Clean up volumes
docker volume prune -f
```

## Test Coverage

The test suite covers:
- ✅ Container build validation
- ✅ Service deployment procedures
- ✅ Health check endpoints
- ✅ Database connectivity
- ✅ Service discovery
- ✅ Horizontal scaling
- ✅ Monitoring configuration
- ✅ Alert routing
- ✅ Failover scenarios
- ✅ Performance baselines
- ✅ Security configurations
- ✅ Data persistence

## Contributing

When adding new deployment features:
1. Add corresponding tests to appropriate test files
2. Update test documentation
3. Ensure tests pass in CI/CD pipeline
4. Add any new test dependencies to requirements.txt