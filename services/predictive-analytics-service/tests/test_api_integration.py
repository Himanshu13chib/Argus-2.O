"""
Integration tests for Predictive Analytics Service API
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from datetime import datetime
import json

from main import app


class TestPredictiveAnalyticsAPI:
    """Test the predictive analytics API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "predictive-analytics"
        assert "timestamp" in data
    
    def test_analyze_patterns_endpoint(self, client):
        """Test pattern analysis endpoint"""
        request_data = {
            "camera_id": "camera_test_1",
            "days_back": 14
        }
        
        response = client.post("/analyze/patterns", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "pattern_type" in data
        assert "frequency" in data
        assert "peak_times" in data
        assert "seasonal_trends" in data
        assert "correlation_factors" in data
        
        # Validate data types
        assert isinstance(data["peak_times"], list)
        assert isinstance(data["seasonal_trends"], dict)
        assert isinstance(data["correlation_factors"], dict)
    
    def test_detect_anomalies_endpoint(self, client):
        """Test anomaly detection endpoint"""
        request_data = {
            "recent_crossings": [
                {
                    "timestamp": "2024-01-01T08:00:00",
                    "confidence": 0.85,
                    "group_size": 1,
                    "speed": 1.2,
                    "direction_consistency": 0.95
                },
                {
                    "timestamp": "2024-01-01T08:30:00",
                    "confidence": 0.90,
                    "group_size": 2,
                    "speed": 1.5,
                    "direction_consistency": 0.88
                }
            ]
        }
        
        response = client.post("/detect/anomalies", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "is_anomaly" in data
        assert "anomaly_score" in data
        assert "anomaly_type" in data
        assert "description" in data
        assert "severity" in data
        assert "timestamp" in data
        
        # Validate data types
        assert isinstance(data["is_anomaly"], bool)
        assert isinstance(data["anomaly_score"], (int, float))
        assert data["severity"] in ["low", "medium", "high"]
    
    def test_detect_anomalies_empty_data(self, client):
        """Test anomaly detection with empty data"""
        request_data = {
            "recent_crossings": []
        }
        
        response = client.post("/detect/anomalies", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["is_anomaly"] is False
        assert data["anomaly_score"] == 0.0
        assert data["anomaly_type"] == "no_data"
    
    def test_adapt_sensitivity_endpoint(self, client):
        """Test sensitivity adaptation endpoint"""
        request_data = {
            "camera_id": "camera_test_2",
            "environmental_data": {
                "lighting_level": 0.6,
                "weather": "clear",
                "temperature": 22.0,
                "wind_speed": 8.0,
                "visibility": 0.9
            },
            "historical_performance": {
                "false_positive_rate": 0.015,
                "detection_accuracy": 0.92,
                "missed_detections": 0.08
            }
        }
        
        response = client.post("/adapt/sensitivity", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["camera_id"] == "camera_test_2"
        assert "original_sensitivity" in data
        assert "adjusted_sensitivity" in data
        assert "confidence_threshold" in data
        assert "adjustments" in data
        assert "environmental_factors" in data
        assert "performance_metrics" in data
        assert "timestamp" in data
        
        # Validate ranges
        assert 0.1 <= data["adjusted_sensitivity"] <= 1.0
        assert 0.5 <= data["confidence_threshold"] <= 0.95
    
    def test_adapt_sensitivity_adverse_conditions(self, client):
        """Test sensitivity adaptation under adverse conditions"""
        request_data = {
            "camera_id": "camera_test_3",
            "environmental_data": {
                "lighting_level": 0.1,  # Very low light
                "weather": "fog",
                "temperature": -2.0,    # Cold
                "wind_speed": 30.0,     # High wind
                "visibility": 0.2       # Very poor visibility
            },
            "historical_performance": {
                "false_positive_rate": 0.005,  # Low false positives
                "detection_accuracy": 0.80,
                "missed_detections": 0.20       # High missed detections
            }
        }
        
        response = client.post("/adapt/sensitivity", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        # Should increase sensitivity due to adverse conditions
        assert data["adjusted_sensitivity"] > data["original_sensitivity"]
        
        # Check individual adjustments
        adjustments = data["adjustments"]
        assert adjustments["lighting"] > 0      # Increase for low light
        assert adjustments["weather"] > 0       # Increase for fog
        assert adjustments["temperature"] > 0   # Increase for extreme temp
        assert adjustments["visibility"] > 0    # Increase for poor visibility
        assert adjustments["performance"] > 0   # Increase for missed detections
    
    def test_predict_maintenance_endpoint(self, client):
        """Test maintenance prediction endpoint"""
        request_data = {
            "hardware_metrics": {
                "cpu_usage": 0.65,
                "memory_usage": 0.70,
                "disk_usage": 0.45,
                "temperature": 45.0,
                "uptime_hours": 168.0,  # 1 week
                "error_rate": 0.005,
                "network_latency": 25.0
            }
        }
        
        response = client.post("/predict/maintenance", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["prediction_type"] == "hardware_maintenance"
        assert "confidence" in data
        assert "predicted_value" in data
        assert "risk_level" in data
        assert "recommendations" in data
        assert "timestamp" in data
        
        # Validate data types and ranges
        assert isinstance(data["confidence"], (int, float))
        assert 0.0 <= data["predicted_value"] <= 1.0
        assert data["risk_level"] in ["minimal", "low", "medium", "high"]
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) > 0
    
    def test_predict_maintenance_critical_system(self, client):
        """Test maintenance prediction for critical system state"""
        request_data = {
            "hardware_metrics": {
                "cpu_usage": 0.95,      # Critical CPU usage
                "memory_usage": 0.98,   # Critical memory usage
                "disk_usage": 0.90,
                "temperature": 75.0,    # High temperature
                "uptime_hours": 2000.0, # Very long uptime
                "error_rate": 0.08,     # High error rate
                "network_latency": 200.0
            }
        }
        
        response = client.post("/predict/maintenance", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        # Should indicate high maintenance need
        assert data["predicted_value"] > 0.5
        assert data["risk_level"] in ["medium", "high"]
        
        # Should have urgent recommendations
        recommendations_text = " ".join(data["recommendations"]).lower()
        assert any(word in recommendations_text for word in ["immediate", "urgent", "critical"])
    
    def test_get_model_status_endpoint(self, client):
        """Test model status endpoint"""
        response = client.get("/models/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert "anomaly_detectors" in data
        assert "scalers" in data
        assert "historical_data" in data
        assert "timestamp" in data
        
        # Validate expected models are present
        assert "crossing_prediction" in data["models"]
        assert "hardware_failure" in data["models"]
        assert "crossing_patterns" in data["anomaly_detectors"]
        assert "behavioral" in data["anomaly_detectors"]
    
    def test_invalid_request_data(self, client):
        """Test API with invalid request data"""
        # Test missing required fields
        response = client.post("/analyze/patterns", json={})
        assert response.status_code == 422  # Validation error
        
        # Test invalid data types
        response = client.post("/analyze/patterns", json={
            "camera_id": 123,  # Should be string
            "days_back": "invalid"  # Should be int
        })
        assert response.status_code == 422
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.post("/analyze/patterns", json={
                "camera_id": f"camera_concurrent_{threading.current_thread().ident}",
                "days_back": 7
            })
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__])