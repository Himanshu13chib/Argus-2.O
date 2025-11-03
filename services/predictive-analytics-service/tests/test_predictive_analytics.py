"""
Tests for Predictive Analytics Service
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from main import PredictiveAnalyticsEngine, analytics_engine
from main import PatternAnalysis, AnomalyDetection, PredictionResult


class TestPredictiveAnalyticsEngine:
    """Test the predictive analytics engine"""
    
    @pytest.fixture
    def engine(self):
        """Create a test analytics engine"""
        return PredictiveAnalyticsEngine()
    
    @pytest.mark.asyncio
    async def test_analyze_historical_patterns_with_data(self, engine):
        """Test historical pattern analysis with mock data"""
        # Test with sufficient data
        result = await engine.analyze_historical_patterns("camera_1", days_back=7)
        
        assert isinstance(result, PatternAnalysis)
        assert result.pattern_type == "temporal_analysis"
        assert result.frequency in ["low", "medium", "high"]
        assert isinstance(result.peak_times, list)
        assert isinstance(result.seasonal_trends, dict)
        assert isinstance(result.correlation_factors, dict)
        
        # Check seasonal trends structure
        expected_trends = ['morning_activity', 'afternoon_activity', 'evening_activity', 'night_activity']
        for trend in expected_trends:
            assert trend in result.seasonal_trends
            assert isinstance(result.seasonal_trends[trend], float)
    
    @pytest.mark.asyncio
    async def test_analyze_historical_patterns_no_data(self, engine):
        """Test historical pattern analysis with no data"""
        # Mock empty data
        with patch.object(engine, '_get_historical_crossings') as mock_get:
            mock_get.return_value = pd.DataFrame()
            
            result = await engine.analyze_historical_patterns("camera_empty", days_back=7)
            
            assert result.pattern_type == "insufficient_data"
            assert result.frequency == "unknown"
            assert result.peak_times == []
            assert result.seasonal_trends == {}
            assert result.correlation_factors == {}
    
    @pytest.mark.asyncio
    async def test_detect_anomalous_behavior_normal(self, engine):
        """Test anomaly detection with normal crossing patterns"""
        # Normal crossing data
        normal_crossings = [
            {
                'timestamp': '2024-01-01 08:00:00',
                'confidence': 0.85,
                'group_size': 1,
                'speed': 1.2,
                'direction_consistency': 0.95
            },
            {
                'timestamp': '2024-01-01 08:30:00',
                'confidence': 0.90,
                'group_size': 1,
                'speed': 1.1,
                'direction_consistency': 0.92
            }
        ]
        
        result = await engine.detect_anomalous_behavior(normal_crossings)
        
        assert isinstance(result, AnomalyDetection)
        assert isinstance(result.is_anomaly, bool)
        assert isinstance(result.anomaly_score, float)
        assert result.anomaly_type in ["normal", "minor_deviation", "unusual_pattern", "highly_unusual_pattern"]
        assert result.severity in ["low", "medium", "high"]
        assert isinstance(result.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_detect_anomalous_behavior_no_data(self, engine):
        """Test anomaly detection with no data"""
        result = await engine.detect_anomalous_behavior([])
        
        assert result.is_anomaly is False
        assert result.anomaly_score == 0.0
        assert result.anomaly_type == "no_data"
        assert result.severity == "low"
    
    @pytest.mark.asyncio
    async def test_adapt_detection_sensitivity_normal_conditions(self, engine):
        """Test sensitivity adaptation under normal conditions"""
        environmental_data = {
            'lighting_level': 0.7,
            'weather': 'clear',
            'temperature': 25.0,
            'wind_speed': 5.0,
            'visibility': 1.0
        }
        
        historical_performance = {
            'false_positive_rate': 0.01,
            'detection_accuracy': 0.95,
            'missed_detections': 0.05
        }
        
        result = await engine.adapt_detection_sensitivity(
            "camera_1", environmental_data, historical_performance
        )
        
        assert result['camera_id'] == "camera_1"
        assert 0.1 <= result['adjusted_sensitivity'] <= 1.0
        assert 0.5 <= result['confidence_threshold'] <= 0.95
        assert 'adjustments' in result
        assert 'environmental_factors' in result
        assert 'performance_metrics' in result
        assert 'timestamp' in result
    
    @pytest.mark.asyncio
    async def test_adapt_detection_sensitivity_adverse_conditions(self, engine):
        """Test sensitivity adaptation under adverse conditions"""
        environmental_data = {
            'lighting_level': 0.1,  # Low light
            'weather': 'fog',       # Poor weather
            'temperature': -5.0,    # Extreme temperature
            'wind_speed': 25.0,
            'visibility': 0.3       # Poor visibility
        }
        
        historical_performance = {
            'false_positive_rate': 0.001,  # Very low false positives
            'detection_accuracy': 0.85,
            'missed_detections': 0.15      # High missed detections
        }
        
        result = await engine.adapt_detection_sensitivity(
            "camera_2", environmental_data, historical_performance
        )
        
        # Should increase sensitivity due to adverse conditions and missed detections
        assert result['adjusted_sensitivity'] > 0.7  # Higher than base
        assert result['adjustments']['lighting'] > 0
        assert result['adjustments']['weather'] > 0
        assert result['adjustments']['temperature'] > 0
        assert result['adjustments']['visibility'] > 0
        assert result['adjustments']['performance'] > 0
    
    @pytest.mark.asyncio
    async def test_predict_hardware_maintenance_healthy(self, engine):
        """Test hardware maintenance prediction for healthy system"""
        healthy_metrics = {
            'cpu_usage': 0.3,
            'memory_usage': 0.4,
            'disk_usage': 0.2,
            'temperature': 35.0,
            'uptime_hours': 48.0,
            'error_rate': 0.001,
            'network_latency': 10.0
        }
        
        result = await engine.predict_hardware_maintenance(healthy_metrics)
        
        assert isinstance(result, PredictionResult)
        assert result.prediction_type == "hardware_maintenance"
        assert result.confidence > 0
        assert 0.0 <= result.predicted_value <= 1.0
        assert result.risk_level in ["minimal", "low", "medium", "high"]
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_predict_hardware_maintenance_critical(self, engine):
        """Test hardware maintenance prediction for critical system"""
        critical_metrics = {
            'cpu_usage': 0.95,      # Very high CPU
            'memory_usage': 0.98,   # Very high memory
            'disk_usage': 0.85,
            'temperature': 80.0,    # High temperature
            'uptime_hours': 1000.0, # Very long uptime
            'error_rate': 0.1,      # High error rate
            'network_latency': 500.0
        }
        
        result = await engine.predict_hardware_maintenance(critical_metrics)
        
        assert result.predicted_value > 0.5  # Should be high maintenance score
        assert result.risk_level in ["medium", "high"]
        assert any("immediate" in rec.lower() or "urgent" in rec.lower() 
                  for rec in result.recommendations)
    
    @pytest.mark.asyncio
    async def test_get_historical_crossings_mock_data(self, engine):
        """Test historical data retrieval generates reasonable mock data"""
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        data = await engine._get_historical_crossings("camera_test", start_date, end_date)
        
        assert isinstance(data, pd.DataFrame)
        if not data.empty:
            assert 'timestamp' in data.columns
            assert 'camera_id' in data.columns
            assert 'confidence' in data.columns
            assert all(data['camera_id'] == "camera_test")
            assert all(0.7 <= conf <= 0.95 for conf in data['confidence'])


class TestPredictiveAnalyticsIntegration:
    """Integration tests for the predictive analytics system"""
    
    @pytest.mark.asyncio
    async def test_full_analytics_workflow(self):
        """Test complete analytics workflow"""
        engine = PredictiveAnalyticsEngine()
        
        # Step 1: Analyze historical patterns
        patterns = await engine.analyze_historical_patterns("camera_integration", days_back=14)
        assert isinstance(patterns, PatternAnalysis)
        
        # Step 2: Detect anomalies in recent data
        recent_crossings = [
            {
                'timestamp': '2024-01-01 03:00:00',  # Unusual time
                'confidence': 0.95,
                'group_size': 5,  # Large group
                'speed': 3.0,     # Fast movement
                'direction_consistency': 0.6  # Inconsistent direction
            }
        ]
        
        anomaly = await engine.detect_anomalous_behavior(recent_crossings)
        assert isinstance(anomaly, AnomalyDetection)
        
        # Step 3: Adapt sensitivity based on conditions
        environmental_data = {
            'lighting_level': 0.2,
            'weather': 'rain',
            'temperature': 15.0,
            'visibility': 0.7
        }
        
        performance_data = {
            'false_positive_rate': 0.02,
            'detection_accuracy': 0.88,
            'missed_detections': 0.12
        }
        
        adaptation = await engine.adapt_detection_sensitivity(
            "camera_integration", environmental_data, performance_data
        )
        
        assert adaptation['adjusted_sensitivity'] != 0.7  # Should be adjusted
        
        # Step 4: Predict maintenance needs
        hardware_metrics = {
            'cpu_usage': 0.75,
            'memory_usage': 0.65,
            'temperature': 55.0,
            'error_rate': 0.02
        }
        
        maintenance = await engine.predict_hardware_maintenance(hardware_metrics)
        assert isinstance(maintenance, PredictionResult)
    
    def test_analytics_engine_initialization(self):
        """Test that analytics engine initializes correctly"""
        engine = PredictiveAnalyticsEngine()
        
        # Check models are initialized
        assert 'crossing_prediction' in engine.models
        assert 'hardware_failure' in engine.models
        
        # Check anomaly detectors are initialized
        assert 'crossing_patterns' in engine.anomaly_detectors
        assert 'behavioral' in engine.anomaly_detectors
        
        # Check scalers are initialized
        assert 'crossing_features' in engine.scalers
        assert 'hardware_metrics' in engine.scalers
        assert 'environmental' in engine.scalers
        
        # Check historical data storage is initialized
        assert 'crossings' in engine.historical_data
        assert 'hardware_health' in engine.historical_data
        assert 'environmental' in engine.historical_data


if __name__ == "__main__":
    pytest.main([__file__])