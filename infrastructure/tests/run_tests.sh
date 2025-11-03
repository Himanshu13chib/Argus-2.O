#!/bin/bash

# Project Argus Deployment Tests Runner
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "🧪 Running Project Argus Deployment Tests"
echo "Project root: ${PROJECT_ROOT}"

# Change to project root
cd "${PROJECT_ROOT}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv_tests" ]; then
    echo "📦 Creating test virtual environment..."
    python3 -m venv venv_tests
fi

# Activate virtual environment
source venv_tests/bin/activate

# Install test dependencies
echo "📦 Installing test dependencies..."
pip install -r infrastructure/tests/requirements.txt

# Function to run container build tests
run_build_tests() {
    echo "🔨 Running container build tests..."
    pytest infrastructure/tests/test_deployment.py::TestContainerBuilds -v
}

# Function to run Docker Compose tests
run_compose_tests() {
    echo "🐳 Running Docker Compose tests..."
    pytest infrastructure/tests/test_deployment.py::TestDockerCompose -v
}

# Function to run Kubernetes tests
run_k8s_tests() {
    echo "☸️  Running Kubernetes tests..."
    pytest infrastructure/tests/test_deployment.py::TestKubernetesDeployment -v
}

# Function to run scaling tests
run_scaling_tests() {
    echo "📈 Running scaling tests..."
    pytest infrastructure/tests/test_deployment.py::TestScalingCapabilities -v
}

# Function to run monitoring tests
run_monitoring_tests() {
    echo "📊 Running monitoring tests..."
    pytest infrastructure/tests/test_monitoring.py -v
}

# Function to run failover tests
run_failover_tests() {
    echo "🔄 Running failover tests..."
    pytest infrastructure/tests/test_deployment.py::TestFailoverCapabilities -v
}

# Function to cleanup test environment
cleanup() {
    echo "🧹 Cleaning up test environment..."
    
    # Stop any running containers
    docker-compose -f docker-compose.yml -f docker-compose.override.yml down -v --remove-orphans 2>/dev/null || true
    docker-compose -f infrastructure/monitoring/docker-compose.monitoring.yml down -v --remove-orphans 2>/dev/null || true
    
    # Remove test images
    docker images -q "project-argus/*:test" | xargs -r docker rmi -f 2>/dev/null || true
    
    echo "✅ Cleanup completed"
}

# Trap cleanup on exit
trap cleanup EXIT

# Parse command line arguments
case "${1:-all}" in
    "build")
        run_build_tests
        ;;
    "compose")
        run_compose_tests
        ;;
    "k8s")
        run_k8s_tests
        ;;
    "scaling")
        run_scaling_tests
        ;;
    "monitoring")
        run_monitoring_tests
        ;;
    "failover")
        run_failover_tests
        ;;
    "all")
        echo "🚀 Running all deployment tests..."
        
        # Run tests in order
        run_build_tests
        echo ""
        
        run_k8s_tests
        echo ""
        
        run_scaling_tests
        echo ""
        
        run_monitoring_tests
        echo ""
        
        echo "🎉 All deployment tests completed!"
        ;;
    "help")
        echo "Usage: $0 [build|compose|k8s|scaling|monitoring|failover|all|help]"
        echo ""
        echo "Commands:"
        echo "  build      - Test container builds"
        echo "  compose    - Test Docker Compose deployment"
        echo "  k8s        - Test Kubernetes manifests"
        echo "  scaling    - Test scaling capabilities"
        echo "  monitoring - Test monitoring system"
        echo "  failover   - Test failover capabilities"
        echo "  all        - Run all tests (default)"
        echo "  help       - Show this help message"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

echo "✅ Test execution completed"