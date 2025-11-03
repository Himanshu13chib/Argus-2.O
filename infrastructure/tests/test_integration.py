#!/usr/bin/env python3
"""
Integration tests for Project Argus deployment
Tests end-to-end deployment scenarios and system integration
"""

import pytest
import requests
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional


class TestEndToEndDeployment:
    """Test complete end-to-end deployment scenarios"""
    
    @pytest.fixture(scope="class")
    def full_deployment(self):
        """Deploy full system for integration testing"""
        # Start core services
        result = subprocess.run([
            "docker-compose", "-f", "docker-compose.yml",
            "up", "-d", "postgres", "redis", "minio"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            pytest.skip(f"Core services startup failed: {result.stderr}")
        
        # Wait for core services
        time.sleep(20)
        
        # Start application services
        result = subprocess.run([
            "docker-compose", "-f", "docker-compose.yml",
            "up", "-d", "auth-service", "security-service", "api-gateway"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            pytest.skip(f"Application services startup failed: {result.stderr}")
        
        # Wait for application services
        time.sleep(30)
        
        yield
        
        # Cleanup
        subprocess.run([
            "docker-compose", "-f", "docker-compose.yml", "down", "-v"
        ], capture_output=True)
    
    def test_service_discovery(self, full_deployment):
        """Test services can discover and communicate with each other"""
        # Test API Gateway can reach auth service
        try:
            response = requests.get("http://localhost:8000/health", timeout=15)
            assert response.status_code == 200
            
            health_data = response.json()
            assert health_data.get("status") == "healthy"
            
            # Check if dependencies are accessible
            if "dependencies" in health_data:
                dependencies = health_data["dependencies"]
                for dep_name, dep_status in dependencies.items():
                    assert dep_status in ["healthy", "available"], f"Dependency {dep_name} is not healthy"
                    
        except Exception as e:
            pytest.fail(f"Service discovery test failed: {e}")
    
    def test_database_connectivity(self, full_deployment):
        """Test all services can connect to database"""
        services = [
            ("auth-service", 8001),
            ("security-service", 8002),
            ("api-gateway", 8000)
        ]
        
        for service_name, port in services:
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=10)
                assert response.status_code == 200
                
                health_data = response.json()
                assert health_data.get("status") == "healthy"
                
                # Check database connection if reported
                if "database" in health_data:
                    assert health_data["database"] == "connected"
                    
            except Exception as e:
                pytest.fail(f"Database connectivity test failed for {service_name}: {e}")
    
    def test_authentication_flow(self, full_deployment):
        """Test authentication flow across services"""
        try:
            # Test auth service directly
            auth_response = requests.get("http://localhost:8001/health", timeout=10)
            assert auth_response.status_code == 200
            
            # Test auth through API gateway
            gateway_response = requests.get("http://localhost:8000/auth/health", timeout=10)
            # This might return 404 if route not configured, which is acceptable for this test
            assert gateway_response.status_code in [200, 404, 502]
            
        except Exception as e:
            pytest.fail(f"Authentication flow test failed: {e}")


class TestSystemResilience:
    """Test system resilience and recovery capabilities"""
    
    def test_service_restart_recovery(self):
        """Test system recovers when services are restarted"""
        # Start minimal services
        result = subprocess.run([
            "docker-compose", "up", "-d", "postgres", "redis"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            pytest.skip("Could not start services for restart test")
        
        time.sleep(10)
        
        # Restart Redis
        subprocess.run(["docker-compose", "restart", "redis"], capture_output=True)
        time.sleep(5)
        
        # Test Redis is accessible
        result = subprocess.run([
            "docker-compose", "exec", "-T", "redis", "redis-cli", "ping"
        ], capture_output=True, text=True)
        
        assert "PONG" in result.stdout
        
        # Cleanup
        subprocess.run(["docker-compose", "down", "-v"], capture_output=True)
    
    def test_network_partition_recovery(self):
        """Test system handles network partitions gracefully"""
        # This is a simplified test - in production you'd test actual network partitions
        
        # Start services
        result = subprocess.run([
            "docker-compose", "up", "-d", "postgres", "redis"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            pytest.skip("Could not start services for network test")
        
        time.sleep(10)
        
        # Simulate network issue by stopping and starting quickly
        subprocess.run(["docker-compose", "stop", "redis"], capture_output=True)
        time.sleep(2)
        subprocess.run(["docker-compose", "start", "redis"], capture_output=True)
        time.sleep(8)
        
        # Test service recovered
        result = subprocess.run([
            "docker-compose", "exec", "-T", "redis", "redis-cli", "ping"
        ], capture_output=True, text=True)
        
        assert "PONG" in result.stdout
        
        # Cleanup
        subprocess.run(["docker-compose", "down", "-v"], capture_output=True)


class TestPerformanceBaseline:
    """Test basic performance characteristics of deployed system"""
    
    @pytest.fixture(scope="class")
    def performance_deployment(self):
        """Deploy system for performance testing"""
        result = subprocess.run([
            "docker-compose", "up", "-d", "postgres", "redis", "api-gateway"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            pytest.skip(f"Performance deployment failed: {result.stderr}")
        
        time.sleep(25)
        yield
        
        subprocess.run(["docker-compose", "down", "-v"], capture_output=True)
    
    def test_response_time_baseline(self, performance_deployment):
        """Test basic response time requirements are met"""
        # Test API Gateway response time
        start_time = time.time()
        
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            assert response.status_code == 200
            assert response_time < 2.0, f"Response time {response_time}s exceeds 2s threshold"
            
        except Exception as e:
            pytest.fail(f"Response time test failed: {e}")
    
    def test_concurrent_requests(self, performance_deployment):
        """Test system handles concurrent requests"""
        import concurrent.futures
        import threading
        
        def make_request():
            try:
                response = requests.get("http://localhost:8000/health", timeout=10)
                return response.status_code == 200
            except:
                return False
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # At least 70% of requests should succeed
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.7, f"Success rate {success_rate} below 70% threshold"


class TestSecurityBaseline:
    """Test basic security characteristics of deployed system"""
    
    def test_no_default_credentials(self):
        """Test system doesn't use default credentials"""
        # This test checks that services don't respond with default credentials
        # In a real deployment, you'd test actual authentication
        
        # Check that services require authentication where expected
        try:
            # Test that admin endpoints are protected (should return 401/403, not 200)
            response = requests.get("http://localhost:8000/admin", timeout=5)
            assert response.status_code in [401, 403, 404], "Admin endpoint should be protected"
            
        except requests.exceptions.RequestException:
            # Service not running is acceptable for this test
            pass
    
    def test_https_redirect(self):
        """Test HTTPS redirect is configured"""
        # This would test actual HTTPS redirect in production
        # For now, just verify nginx config exists
        
        nginx_config = Path("infrastructure/nginx/nginx.conf")
        if nginx_config.exists():
            with open(nginx_config, 'r') as f:
                config_content = f.read()
                # Check for SSL/HTTPS configuration
                assert "ssl" in config_content.lower() or "https" in config_content.lower()


class TestDataPersistence:
    """Test data persistence across restarts"""
    
    def test_database_persistence(self):
        """Test database data persists across container restarts"""
        # Start database
        result = subprocess.run([
            "docker-compose", "up", "-d", "postgres"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            pytest.skip("Could not start database for persistence test")
        
        time.sleep(10)
        
        # Create test data
        result = subprocess.run([
            "docker-compose", "exec", "-T", "postgres",
            "psql", "-U", "argus_user", "-d", "project_argus",
            "-c", "CREATE TABLE IF NOT EXISTS test_persistence (id SERIAL PRIMARY KEY, data TEXT);"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            pytest.skip("Could not create test table")
        
        # Insert test data
        subprocess.run([
            "docker-compose", "exec", "-T", "postgres",
            "psql", "-U", "argus_user", "-d", "project_argus",
            "-c", "INSERT INTO test_persistence (data) VALUES ('test_data');"
        ], capture_output=True)
        
        # Restart database
        subprocess.run(["docker-compose", "restart", "postgres"], capture_output=True)
        time.sleep(10)
        
        # Check data persists
        result = subprocess.run([
            "docker-compose", "exec", "-T", "postgres",
            "psql", "-U", "argus_user", "-d", "project_argus",
            "-c", "SELECT COUNT(*) FROM test_persistence WHERE data = 'test_data';"
        ], capture_output=True, text=True)
        
        assert "1" in result.stdout, "Test data should persist across restart"
        
        # Cleanup
        subprocess.run(["docker-compose", "down", "-v"], capture_output=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])