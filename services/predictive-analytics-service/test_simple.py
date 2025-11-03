"""
Simple test to verify the predictive analytics implementation
"""

import asyncio
from datetime import datetime
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the shared modules to avoid import errors
class MockSecurityManager:
    pass

class MockDetection:
    pass

class MockAlert:
    pass

class MockTrack:
    pass

# Mock the shared modules
sys.modules['shared.interfaces.security'] = type('module', (), {'SecurityManager': MockSecurityManager})()
sys.modules['shared.models.detection'] = type('module', (), {'Detection': MockDetection})()
sys.modules['shared.models.alerts'] = type('module', (), {'Alert': MockAlert})()
sys.modules['shared.models.tracking'] = type('module', (), {'Track': MockTrack})()

# Now import the main module
from main import PredictiveAnalyticsEngine, analytics_engine

async def test_analytics_engine():
    """Test the analytics engine functionality"""
    print("Testing Predictive Analytics Engine...")
    
    # Test 1: Initialize engine
    engine = PredictiveAnalyticsEngine()
    print("✓ Engine initialized successfully")
    
    # Test 2: Analyze historical patterns
    patterns = await engine.analyze_historical_patterns("test_camera", days_back=7)
    print(f"✓ Pattern analysis completed: {patterns.pattern_type}")
    
    # Test 3: Detect anomalies
    test_crossings = [
        {
            'timestamp': '2024-01-01 08:00:00',
            'confidence': 0.85,
            'group_size': 1,
            'speed': 1.2,
            'direction_consistency': 0.95
        }
    ]
    
    anomaly = await engine.detect_anomalous_behavior(test_crossings)
    print(f"✓ Anomaly detection completed: {anomaly.anomaly_type}")
    
    # Test 4: Adapt sensitivity
    env_data = {
        'lighting_level': 0.6,
        'weather': 'clear',
        'temperature': 25.0,
        'visibility': 0.9
    }
    
    perf_data = {
        'false_positive_rate': 0.01,
        'detection_accuracy': 0.95,
        'missed_detections': 0.05
    }
    
    adaptation = await engine.adapt_detection_sensitivity("test_camera", env_data, perf_data)
    print(f"✓ Sensitivity adaptation completed: {adaptation['adjusted_sensitivity']:.3f}")
    
    # Test 5: Predict maintenance
    hardware_metrics = {
        'cpu_usage': 0.65,
        'memory_usage': 0.70,
        'temperature': 45.0,
        'error_rate': 0.005
    }
    
    maintenance = await engine.predict_hardware_maintenance(hardware_metrics)
    print(f"✓ Maintenance prediction completed: {maintenance.risk_level}")
    
    print("\nAll tests passed! ✓")

if __name__ == "__main__":
    asyncio.run(test_analytics_engine())