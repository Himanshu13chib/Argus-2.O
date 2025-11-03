#!/usr/bin/env python3
"""
Deployment tests for Project Argus infrastructure
Tests container builds, deployment procedures, scaling, and monitoring
"""

import pytest
import docker
import requests
import time
import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional


class TestContainerBuilds:
    """Test Docker container builds and basic functionality"""
    
    @pytest.fixture(scope="class")
    def docker_client(self):
        """Docker client fixture"""
        return docker.from_env()
    
    def test_edge_container_build(self, docker_client):
        """Test edge node container builds successfully"""
        try:
            # Build edge container
            image, logs = docker_client.images.build(
                path=".",
                dockerfile="edge/Dockerfile",
                tag="project-argus/edge:test",
                target="production"
            )
            assert image is not None
            assert "project-argus/edge:test" in [tag for tag in image.tags]
        except Exception as e:
            pytest.fail(f"Edge container build failed: {e}")
    
    def test_dashboard_container_build(self, docker_client):
        """Test dashboard container builds successfully"""
        try:
            image, logs = docker_client.images.build(
                path=".",
                dockerfile="dashboard/Dockerfile",
                tag="project-argus/dashboard:test",
                target="production"
            )
            assert image is not None
            assert "project-argus/dashboard:test" in [tag for tag in image.tags]
        except Exception as e:
            pytest.fail(f"Dashboard container build failed: {e}")
    
    def test_api_gateway_container_build(self, docker_client):
        """Test API gateway container builds successfully"""
        try:
            image, logs = docker_client.images.build(
                path=".",
                dockerfile="services/api-gateway/Dockerfile",
                tag="project-argus/api-gateway:test",
                target="production"
            )
            assert image is not None
            assert "project-argus/api-gateway:test" in [tag for tag in image.tags]
        except Exception as e:
            pytest.fail(f"API gateway container build failed: {e}")
    
    def test_service_containers_build(self, docker_client):
        """Test all service containers build successfully"""
        services = [
            "auth-service",
            "security-service", 
            "alert-service",
            "tracking-service",
            "evidence-service"
        ]
        
        for service in services:
            try:
                image, logs = docker_client.images.build(
                    path=".",
                    dockerfile=f"services/{service}/Dockerfile",
                    tag=f"project-argus/{service}:test",
                    target="production"
                )
                assert image is not None
                assert f"project-argus/{service}:test" in [tag for tag in image.tags]
            except Exception as e:
                pytest.fail(f"{service} container build failed: {e}")


class TestDockerCompose:
    """Test Docker Compose deployment procedures"""
    
    @pytest.fixture(scope="class")
    def compose_project(self):
        """Start Docker Compose project for testing"""
        # Start services
        result = subprocess.run([
            "docker-compose", "-f", "docker-compose.yml", 
            "-f", "docker-compose.override.yml",
            "up", "-d", "--build"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            pytest.fail(f"Docker Compose startup failed: {result.stderr}")
        
        # Wait for services to be ready
        time.sleep(30)
        
        yield
        
        # Cleanup
        subprocess.run([
            "docker-compose", "-f", "docker-compose.yml",
            "-f", "docker-compose.override.yml", 
            "down", "-v"
        ], capture_output=True)
    
    def test_database_health(self, compose_project):
        """Test database services are healthy"""
        # Test PostgreSQL
        result = subprocess.run([
            "docker-compose", "exec", "-T", "postgres",
            "pg_isready", "-U", "argus_user", "-d", "project_argus"
        ], capture_output=True, text=True)
        assert result.returncode == 0
        
        # Test Redis
        result = subprocess.run([
            "docker-compose", "exec", "-T", "redis",
            "redis-cli", "ping"
        ], capture_output=True, text=True)
        assert "PONG" in result.stdout
    
    def test_service_health_endpoints(self, compose_project):
        """Test all services respond to health checks"""
        services = {
            "api-gateway": 8000,
            "auth-service": 8001,
            "security-service": 8002,
            "alert-service": 8003,
            "tracking-service": 8004,
            "evidence-service": 8005
        }
        
        for service, port in services.items():
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=10)
                assert response.status_code == 200
                health_data = response.json()
                assert health_data.get("status") == "healthy"
            except Exception as e:
                pytest.fail(f"{service} health check failed: {e}")
    
    def test_dashboard_accessibility(self, compose_project):
        """Test dashboard is accessible"""
        try:
            response = requests.get("http://localhost:3000", timeout=10)
            assert response.status_code == 200
            assert "text/html" in response.headers.get("content-type", "")
        except Exception as e:
            pytest.fail(f"Dashboard accessibility test failed: {e}")


class TestKubernetesDeployment:
    """Test Kubernetes deployment procedures"""
    
    @pytest.fixture(scope="class")
    def kubectl_available(self):
        """Check if kubectl is available"""
        result = subprocess.run(["kubectl", "version", "--client"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            pytest.skip("kubectl not available")
        return True
    
    def test_kubernetes_manifests_valid(self, kubectl_available):
        """Test Kubernetes manifests are valid"""
        manifest_files = [
            "infrastructure/k8s/namespace.yaml",
            "infrastructure/k8s/configmap.yaml",
            "infrastructure/k8s/secrets.yaml",
            "infrastructure/k8s/storage.yaml",
            "infrastructure/k8s/database.yaml",
            "infrastructure/k8s/redis.yaml",
            "infrastructure/k8s/minio.yaml",
            "infrastructure/k8s/services.yaml",
            "infrastructure/k8s/api-gateway.yaml",
            "infrastructure/k8s/microservices.yaml",
            "infrastructure/k8s/dashboard.yaml",
            "infrastructure/k8s/ingress.yaml"
        ]
        
        for manifest_file in manifest_files:
            if Path(manifest_file).exists():
                result = subprocess.run([
                    "kubectl", "apply", "--dry-run=client", "-f", manifest_file
                ], capture_output=True, text=True)
                assert result.returncode == 0, f"Invalid manifest: {manifest_file}\n{result.stderr}"
    
    def test_hpa_configuration(self, kubectl_available):
        """Test Horizontal Pod Autoscaler configurations"""
        hpa_files = [
            "infrastructure/k8s/api-gateway.yaml",
            "infrastructure/k8s/dashboard.yaml",
            "infrastructure/k8s/load-balancer.yaml"
        ]
        
        for hpa_file in hpa_files:
            if Path(hpa_file).exists():
                with open(hpa_file, 'r') as f:
                    content = f.read()
                    # Check for HPA configuration
                    if "HorizontalPodAutoscaler" in content:
                        docs = yaml.safe_load_all(content)
                        for doc in docs:
                            if doc and doc.get("kind") == "HorizontalPodAutoscaler":
                                spec = doc.get("spec", {})
                                assert "minReplicas" in spec
                                assert "maxReplicas" in spec
                                assert spec["minReplicas"] < spec["maxReplicas"]
                                assert "metrics" in spec


class TestScalingCapabilities:
    """Test horizontal scaling and load balancing"""
    
    def test_docker_compose_scaling(self):
        """Test Docker Compose service scaling"""
        # Scale API gateway to 3 replicas
        result = subprocess.run([
            "docker-compose", "up", "-d", "--scale", "api-gateway=3"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Check if 3 instances are running
            result = subprocess.run([
                "docker-compose", "ps", "api-gateway"
            ], capture_output=True, text=True)
            
            # Count running instances
            running_instances = result.stdout.count("Up")
            assert running_instances >= 2, "Scaling test requires at least 2 instances"
    
    def test_load_balancer_configuration(self):
        """Test load balancer configuration is valid"""
        lb_file = "infrastructure/k8s/load-balancer.yaml"
        if Path(lb_file).exists():
            with open(lb_file, 'r') as f:
                content = yaml.safe_load_all(f.read())
                
                for doc in content:
                    if doc and doc.get("kind") == "Service":
                        spec = doc.get("spec", {})
                        if spec.get("type") == "LoadBalancer":
                            assert "ports" in spec
                            assert len(spec["ports"]) > 0
                            assert "selector" in spec


class TestMonitoringSystem:
    """Test monitoring and alerting system functionality"""
    
    @pytest.fixture(scope="class")
    def monitoring_stack(self):
        """Start monitoring stack for testing"""
        result = subprocess.run([
            "docker-compose", "-f", "infrastructure/monitoring/docker-compose.monitoring.yml",
            "up", "-d"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            pytest.skip(f"Monitoring stack startup failed: {result.stderr}")
        
        # Wait for services to be ready
        time.sleep(45)
        
        yield
        
        # Cleanup
        subprocess.run([
            "docker-compose", "-f", "infrastructure/monitoring/docker-compose.monitoring.yml",
            "down", "-v"
        ], capture_output=True)
    
    def test_prometheus_accessibility(self, monitoring_stack):
        """Test Prometheus is accessible and collecting metrics"""
        try:
            # Test Prometheus UI
            response = requests.get("http://localhost:9090", timeout=10)
            assert response.status_code == 200
            
            # Test metrics endpoint
            response = requests.get("http://localhost:9090/metrics", timeout=10)
            assert response.status_code == 200
            assert "prometheus_" in response.text
            
            # Test targets endpoint
            response = requests.get("http://localhost:9090/api/v1/targets", timeout=10)
            assert response.status_code == 200
            targets_data = response.json()
            assert targets_data.get("status") == "success"
            
        except Exception as e:
            pytest.fail(f"Prometheus accessibility test failed: {e}")
    
    def test_grafana_accessibility(self, monitoring_stack):
        """Test Grafana is accessible"""
        try:
            response = requests.get("http://localhost:3001", timeout=10)
            assert response.status_code == 200
            assert "Grafana" in response.text
        except Exception as e:
            pytest.fail(f"Grafana accessibility test failed: {e}")
    
    def test_jaeger_accessibility(self, monitoring_stack):
        """Test Jaeger is accessible"""
        try:
            response = requests.get("http://localhost:16686", timeout=10)
            assert response.status_code == 200
            assert "Jaeger" in response.text
        except Exception as e:
            pytest.fail(f"Jaeger accessibility test failed: {e}")
    
    def test_alertmanager_configuration(self, monitoring_stack):
        """Test AlertManager configuration is valid"""
        try:
            response = requests.get("http://localhost:9093/api/v1/status", timeout=10)
            assert response.status_code == 200
            status_data = response.json()
            assert status_data.get("status") == "success"
        except Exception as e:
            pytest.fail(f"AlertManager configuration test failed: {e}")


class TestFailoverCapabilities:
    """Test system failover and recovery procedures"""
    
    def test_database_failover_simulation(self):
        """Test database failover simulation"""
        # This would require a more complex setup with database clustering
        # For now, test that the system can handle database disconnection gracefully
        
        # Start services
        result = subprocess.run([
            "docker-compose", "up", "-d", "postgres", "redis"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            time.sleep(10)
            
            # Stop database
            subprocess.run([
                "docker-compose", "stop", "postgres"
            ], capture_output=True)
            
            time.sleep(5)
            
            # Restart database
            subprocess.run([
                "docker-compose", "start", "postgres"
            ], capture_output=True)
            
            time.sleep(10)
            
            # Test database is accessible again
            result = subprocess.run([
                "docker-compose", "exec", "-T", "postgres",
                "pg_isready", "-U", "argus_user", "-d", "project_argus"
            ], capture_output=True, text=True)
            
            assert result.returncode == 0
    
    def test_service_recovery(self):
        """Test service recovery after failure"""
        # Start a single service
        result = subprocess.run([
            "docker-compose", "up", "-d", "redis"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            time.sleep(5)
            
            # Kill the service
            subprocess.run([
                "docker-compose", "kill", "redis"
            ], capture_output=True)
            
            time.sleep(2)
            
            # Restart the service
            subprocess.run([
                "docker-compose", "up", "-d", "redis"
            ], capture_output=True)
            
            time.sleep(10)
            
            # Test service is healthy
            result = subprocess.run([
                "docker-compose", "exec", "-T", "redis",
                "redis-cli", "ping"
            ], capture_output=True, text=True)
            
            assert "PONG" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])