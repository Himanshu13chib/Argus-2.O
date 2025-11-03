"""
Advanced Tests for Predictive Analytics System
Tests historical pattern analysis, anomaly detection, and environmental adaptation
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from main import PredictiveAnalyticsService, AnalysisRequest, PredictionRequest
from pattern_analyzer import PatternAnalyzer, CrossingPattern, AnomalyDetection
from environmental_adapter import EnvironmentalAdapter, WeatherCondition, AdaptationResult

class TestPredictiveAnalyticsService:
    """Test predictive analytics service advanced functionality"""
    
    @pytest.fixture
    def analytics_service(self):
        """Create predictive analytics service instance"""
        return PredictiveAnalyticsService()
    
    @pytest.fixture
    def historical_data(self):
        """Create sample historical crossing data"""
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='H')
        data = []
        
        for i, date in enumerate(dates):
            # Simulate daily patterns with some randomness
            hour = date.hour
            day_of_week = date.weekday()
            
            # Higher activity during night hours and weekends
            base_crossings = 2 if 22 <= hour or hour <= 6 else 1
            if day_of_week >= 5:  # Weekend
                base_crossings *= 1.5
            
            crossings = max(0, int(base_crossings + np.random.poisson(1)))
            
            data.append({
                'timestamp': date,
                'crossing_count': crossings,
                'hour': hour,
                'day_of_week': day_of_week,
                'weather_condition': np.random.choice(['clear', 'cloudy', 'rainy', 'foggy']),
                'temperature': 20 + 10 * np.sin(2 * np.pi * i / (24 * 7)) + np.random.normal(0, 3),
                'location': f"sector-{np.random.randint(1, 6)}"
            })
        
        return pd.DataFrame(data)
    
    @pytest.mark.asyncio
    async def test_historical_pattern_analysis(self, analytics_service, historical_data):
        """Test historical pattern analysis"""
        analysis_request = AnalysisRequest(
            analysis_type="historical_patterns",
            time_range={
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-03-31T23:59:59Z"
            },
            location="sector-1",
            parameters={
                "pattern_types": ["hourly", "daily", "weekly"],
                "min_confidence": 0.7
            }
        )
        
        # Mock historical data retrieval
        with patch.object(analytics_service, '_get_historical_data', return_value=historical_data):
            result = await analytics_service.analyze_patterns(analysis_request)
        
        assert result is not None
        assert result.analysis_type == "historical_patterns"
        assert "patterns" in result.results
        assert len(result.results["patterns"]) > 0
        
        # Check for expected pattern types
        pattern_types = [p["type"] for p in result.results["patterns"]]
        assert "hourly" in pattern_types
        assert "daily" in pattern_types
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, analytics_service, historical_data):
        """Test anomaly detection in crossing patterns"""
        # Add some anomalous data points
        anomaly_data = historical_data.copy()
        
        # Insert anomalies - unusually high crossing counts
        anomaly_indices = np.random.choice(len(anomaly_data), 10, replace=False)
        for idx in anomaly_indices:
            anomaly_data.loc[idx, 'crossing_count'] = 20 + np.random.randint(0, 10)
        
        analysis_request = AnalysisRequest(
            analysis_type="anomaly_detection",
            time_range={
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-03-31T23:59:59Z"
            },
            parameters={
                "anomaly_threshold": 2.0,  # Standard deviations
                "min_anomaly_score": 0.8
            }
        )
        
        with patch.object(analytics_service, '_get_historical_data', return_value=anomaly_data):
            result = await analytics_service.analyze_patterns(analysis_request)
        
        assert result is not None
        assert "anomalies" in result.results
        assert len(result.results["anomalies"]) > 0
        
        # Check anomaly properties
        for anomaly in result.results["anomalies"]:
            assert "timestamp" in anomaly
            assert "anomaly_score" in anomaly
            assert anomaly["anomaly_score"] >= 0.8
    
    @pytest.mark.asyncio
    async def test_predictive_modeling(self, analytics_service, historical_data):
        """Test predictive modeling for future crossings"""
        prediction_request = PredictionRequest(
            prediction_type="crossing_forecast",
            time_horizon=24,  # 24 hours
            location="sector-1",
            parameters={
                "model_type": "time_series",
                "confidence_interval": 0.95
            }
        )
        
        with patch.object(analytics_service, '_get_historical_data', return_value=historical_data):
            result = await analytics_service.generate_predictions(prediction_request)
        
        assert result is not None
        assert result.prediction_type == "crossing_forecast"
        assert "predictions" in result.results
        assert len(result.results["predictions"]) == 24  # Hourly predictions
        
        # Check prediction structure
        for prediction in result.results["predictions"]:
            assert "timestamp" in prediction
            assert "predicted_value" in prediction
            assert "confidence_lower" in prediction
            assert "confidence_upper" in prediction
    
    @pytest.mark.asyncio
    async def test_environmental_adaptation(self, analytics_service):
        """Test environmental adaptation of detection sensitivity"""
        adaptation_request = {
            "location": "sector-1",
            "current_conditions": {
                "weather": "foggy",
                "temperature": 5.0,
                "humidity": 95.0,
                "wind_speed": 15.0,
                "visibility": 100.0  # meters
            },
            "historical_performance": {
                "clear_weather_accuracy": 0.95,
                "foggy_weather_accuracy": 0.78
            }
        }
        
        result = await analytics_service.adapt_to_environment(adaptation_request)
        
        assert result is not None
        assert "sensitivity_adjustment" in result
        assert "confidence_threshold" in result
        assert "recommended_settings" in result
        
        # In foggy conditions, sensitivity should be increased
        assert result["sensitivity_adjustment"] > 1.0

class TestPatternAnalyzer:
    """Test pattern analysis algorithms"""
    
    @pytest.fixture
    def pattern_analyzer(self):
        """Create pattern analyzer instance"""
        return PatternAnalyzer()
    
    @pytest.fixture
    def crossing_data(self):
        """Create crossing event data"""
        dates = pd.date_range(start='2024-01-01', end='2024-02-29', freq='H')
        data = []
        
        for date in dates:
            hour = date.hour
            day_of_week = date.weekday()
            
            # Create realistic patterns
            # Higher activity at night (22-06) and weekends
            if 22 <= hour or hour <= 6:
                base_rate = 3.0
            elif 18 <= hour <= 21:
                base_rate = 2.0
            else:
                base_rate = 1.0
            
            if day_of_week >= 5:  # Weekend
                base_rate *= 1.5
            
            crossings = np.random.poisson(base_rate)
            
            data.append({
                'timestamp': date,
                'crossing_count': crossings,
                'hour': hour,
                'day_of_week': day_of_week
            })
        
        return pd.DataFrame(data)
    
    def test_hourly_pattern_detection(self, pattern_analyzer, crossing_data):
        """Test hourly pattern detection"""
        patterns = pattern_analyzer.detect_hourly_patterns(crossing_data)
        
        assert len(patterns) > 0
        
        # Should detect night-time peak
        night_patterns = [p for p in patterns if p.pattern_type == "hourly_peak" and 22 <= p.time_component <= 23 or 0 <= p.time_component <= 6]
        assert len(night_patterns) > 0
        
        # Check pattern properties
        for pattern in patterns:
            assert hasattr(pattern, 'confidence_score')
            assert hasattr(pattern, 'frequency')
            assert 0 <= pattern.confidence_score <= 1
    
    def test_weekly_pattern_detection(self, pattern_analyzer, crossing_data):
        """Test weekly pattern detection"""
        patterns = pattern_analyzer.detect_weekly_patterns(crossing_data)
        
        assert len(patterns) > 0
        
        # Should detect weekend patterns
        weekend_patterns = [p for p in patterns if p.pattern_type == "weekly_peak" and p.time_component >= 5]
        assert len(weekend_patterns) > 0
    
    def test_seasonal_trend_analysis(self, pattern_analyzer):
        """Test seasonal trend analysis"""
        # Create data with seasonal trends
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        data = []
        
        for i, date in enumerate(dates):
            # Simulate seasonal variation
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 365.25)  # Annual cycle
            base_crossings = 5 * seasonal_factor + np.random.normal(0, 1)
            
            data.append({
                'timestamp': date,
                'crossing_count': max(0, int(base_crossings)),
                'month': date.month
            })
        
        seasonal_data = pd.DataFrame(data)
        trends = pattern_analyzer.analyze_seasonal_trends(seasonal_data)
        
        assert len(trends) > 0
        assert any(trend.trend_type == "seasonal" for trend in trends)
    
    def test_anomaly_detection_algorithm(self, pattern_analyzer, crossing_data):
        """Test anomaly detection algorithm"""
        # Add some clear anomalies
        anomaly_data = crossing_data.copy()
        
        # Insert high anomalies
        anomaly_indices = [100, 200, 300]
        for idx in anomaly_indices:
            anomaly_data.loc[idx, 'crossing_count'] = 50  # Very high value
        
        anomalies = pattern_analyzer.detect_anomalies(
            anomaly_data,
            threshold=2.0,
            method="statistical"
        )
        
        assert len(anomalies) >= len(anomaly_indices)
        
        # Check that our inserted anomalies are detected
        detected_indices = [a.data_index for a in anomalies]
        for idx in anomaly_indices:
            assert idx in detected_indices or any(abs(idx - di) <= 2 for di in detected_indices)

class TestEnvironmentalAdapter:
    """Test environmental adaptation algorithms"""
    
    @pytest.fixture
    def environmental_adapter(self):
        """Create environmental adapter instance"""
        return EnvironmentalAdapter()
    
    def test_weather_condition_adaptation(self, environmental_adapter):
        """Test adaptation to different weather conditions"""
        conditions = [
            WeatherCondition(
                weather_type="clear",
                temperature=25.0,
                humidity=45.0,
                visibility=10000.0,
                wind_speed=5.0
            ),
            WeatherCondition(
                weather_type="foggy",
                temperature=8.0,
                humidity=95.0,
                visibility=50.0,
                wind_speed=2.0
            ),
            WeatherCondition(
                weather_type="rainy",
                temperature=15.0,
                humidity=85.0,
                visibility=500.0,
                wind_speed=20.0
            )
        ]
        
        for condition in conditions:
            adaptation = environmental_adapter.adapt_to_weather(condition)
            
            assert isinstance(adaptation, AdaptationResult)
            assert adaptation.sensitivity_multiplier > 0
            assert 0 <= adaptation.confidence_threshold <= 1
            
            # Foggy conditions should increase sensitivity
            if condition.weather_type == "foggy":
                assert adaptation.sensitivity_multiplier > 1.0
            
            # Clear conditions should have standard settings
            if condition.weather_type == "clear":
                assert abs(adaptation.sensitivity_multiplier - 1.0) < 0.1
    
    def test_time_based_adaptation(self, environmental_adapter):
        """Test time-based adaptation"""
        times = [
            datetime(2024, 6, 15, 12, 0),  # Noon
            datetime(2024, 6, 15, 2, 0),   # Night
            datetime(2024, 12, 15, 12, 0), # Winter noon
            datetime(2024, 12, 15, 2, 0)   # Winter night
        ]
        
        for time in times:
            adaptation = environmental_adapter.adapt_to_time(time, latitude=28.6139)
            
            assert isinstance(adaptation, AdaptationResult)
            
            # Night time should have different settings
            if time.hour < 6 or time.hour > 20:
                assert adaptation.thermal_weight > adaptation.visible_weight
    
    def test_historical_performance_adaptation(self, environmental_adapter):
        """Test adaptation based on historical performance"""
        performance_data = {
            "clear": {"accuracy": 0.95, "false_positive_rate": 0.02},
            "cloudy": {"accuracy": 0.92, "false_positive_rate": 0.03},
            "rainy": {"accuracy": 0.88, "false_positive_rate": 0.05},
            "foggy": {"accuracy": 0.75, "false_positive_rate": 0.08}
        }
        
        current_condition = WeatherCondition(
            weather_type="foggy",
            temperature=5.0,
            humidity=95.0,
            visibility=100.0,
            wind_speed=10.0
        )
        
        adaptation = environmental_adapter.adapt_based_on_performance(
            current_condition,
            performance_data
        )
        
        assert isinstance(adaptation, AdaptationResult)
        
        # Poor historical performance should increase sensitivity
        assert adaptation.sensitivity_multiplier > 1.0
        
        # Should recommend additional measures for poor conditions
        assert len(adaptation.recommended_actions) > 0

class TestPredictiveMaintenanceIntegration:
    """Test predictive maintenance for hardware components"""
    
    @pytest.fixture
    def analytics_service(self):
        """Create analytics service for maintenance testing"""
        return PredictiveAnalyticsService()
    
    @pytest.mark.asyncio
    async def test_hardware_health_prediction(self, analytics_service):
        """Test hardware health prediction"""
        # Mock hardware sensor data
        sensor_data = {
            "camera_001": {
                "temperature": [45.2, 46.1, 47.3, 48.5, 49.2],  # Rising temperature
                "vibration": [0.1, 0.12, 0.15, 0.18, 0.22],     # Increasing vibration
                "power_consumption": [12.5, 12.8, 13.2, 13.8, 14.5],  # Rising power
                "timestamps": [
                    datetime.now() - timedelta(hours=4),
                    datetime.now() - timedelta(hours=3),
                    datetime.now() - timedelta(hours=2),
                    datetime.now() - timedelta(hours=1),
                    datetime.now()
                ]
            }
        }
        
        maintenance_request = {
            "component_type": "camera",
            "component_id": "camera_001",
            "sensor_data": sensor_data["camera_001"],
            "prediction_horizon": 168  # 1 week in hours
        }
        
        result = await analytics_service.predict_maintenance_needs(maintenance_request)
        
        assert result is not None
        assert "health_score" in result
        assert "predicted_failure_time" in result
        assert "maintenance_recommendations" in result
        
        # Rising temperature and vibration should indicate declining health
        assert result["health_score"] < 0.9  # Should be concerning
        
        # Should have maintenance recommendations
        assert len(result["maintenance_recommendations"]) > 0
    
    @pytest.mark.asyncio
    async def test_component_lifecycle_analysis(self, analytics_service):
        """Test component lifecycle analysis"""
        # Mock component usage data
        usage_data = {
            "component_id": "drone_motor_001",
            "component_type": "motor",
            "installation_date": datetime.now() - timedelta(days=365),
            "usage_hours": 2400,
            "maintenance_history": [
                {"date": datetime.now() - timedelta(days=180), "type": "routine"},
                {"date": datetime.now() - timedelta(days=90), "type": "repair"}
            ],
            "performance_metrics": {
                "efficiency": [0.95, 0.93, 0.91, 0.88, 0.85],  # Declining
                "noise_level": [35, 37, 39, 42, 45],            # Increasing
                "timestamps": [
                    datetime.now() - timedelta(days=120),
                    datetime.now() - timedelta(days=90),
                    datetime.now() - timedelta(days=60),
                    datetime.now() - timedelta(days=30),
                    datetime.now()
                ]
            }
        }
        
        result = await analytics_service.analyze_component_lifecycle(usage_data)
        
        assert result is not None
        assert "remaining_useful_life" in result
        assert "replacement_recommendation" in result
        assert "cost_analysis" in result
        
        # Declining performance should suggest replacement soon
        assert result["remaining_useful_life"] < 365  # Less than a year

if __name__ == "__main__":
    pytest.main([__file__, "-v"])