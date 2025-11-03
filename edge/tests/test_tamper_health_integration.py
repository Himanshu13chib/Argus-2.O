"""
Integration tests for Tamper Detection and Health Monitoring systems.

Tests the integration between tamper detection and health monitoring
to ensure they work together effectively for comprehensive system monitoring.
"""

import pytest
import numpy as np
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from edge.src.tamper_detector import TamperDetector, TamperType
from edge.src.health_monitor import HealthMonitor
from shared.models.health import ComponentType, ComponentStatus


class TestTamperHealthIntegration:
    """Integration tests for tamper detection and health monitoring."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.health_monitor = HealthMonitor(check_interval=0.1)
        self.tamper_detector = TamperDetector(camera_id="integration_camera", baseline_frames=3)
        
        # Register camera with health monitor
        self.health_monitor.register_component(
            component_id="integration_camera",
            component_type=ComponentType.CAMERA
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.health_monitor.is_running:
            self.health_monitor.stop_monitoring()
    
    def test_tamper_detection_health_reporting(self):
        """Test that tamper detection integrates with health monitoring."""
        # Establish baseline for tamper detector
        normal_frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.tamper_detector.establish_baseline(normal_frame)
        
        # Register tamper detector health check with health monitor
        def tamper_health_check():
            status = self.tamper_detector.get_detection_status()
            return {
                "baseline_established": status["baseline_established"],
                "recent_events_count": status["recent_events_count"],
                "consecutive_failures": status["consecutive_failures"],
                "avg_brightness": status["baseline_stats"]["avg_brightness"]
            }
        
        self.health_monitor.register_health_check("integration_camera", tamper_health_check)
        
        # Run health check
        self.health_monitor.run_health_checks()
        
        # Verify health data was updated
        component_health = self.health_monitor.get_component_health("integration_camera")
        assert component_health is not None
        assert component_health.metadata["baseline_established"] is True
        assert component_health.metadata["recent_events_count"] == 0
    
    def test_tamper_event_affects_health_status(self):
        """Test that tamper events affect component health status."""
        # Establish baseline
        normal_frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.tamper_detector.establish_baseline(normal_frame)
        
        # Create health check that reports tamper events
        def tamper_aware_health_check():
            # Simulate tamper detection
            dark_frame = np.random.randint(0, 30, (480, 640, 3), dtype=np.uint8)
            events = self.tamper_detector.detect_tampering(dark_frame)
            
            # Report health based on tamper events
            health_data = {
                "cpu_usage": 50.0,
                "memory_usage": 60.0,
                "tamper_events_count": len(events)
            }
            
            # If tamper events detected, report as unhealthy
            if events:
                health_data["error_messages"] = [f"Tamper detected: {events[0].description}"]
            
            return health_data
        
        self.health_monitor.register_health_check("integration_camera", tamper_aware_health_check)
        
        # Run health check
        self.health_monitor.run_health_checks()
        
        # Check that tamper events affected health status
        component_health = self.health_monitor.get_component_health("integration_camera")
        assert component_health is not None
        
        if component_health.metadata.get("tamper_events_count", 0) > 0:
            # Should have error messages if tamper detected
            assert len(component_health.error_messages) > 0
            assert "tamper" in component_health.error_messages[0].lower()
    
    @patch('socket.create_connection')
    def test_network_tamper_and_health_correlation(self, mock_socket):
        """Test correlation between network tampering and health monitoring."""
        # Simulate network failure
        mock_socket.side_effect = OSError("Network unreachable")
        
        # Check system tampering (should detect network issues)
        system_events = self.tamper_detector.detect_system_tampering()
        network_events = [e for e in system_events if e.tamper_type == TamperType.CABLE_CUT]
        
        # Register health check that considers network status
        def network_aware_health_check():
            is_network_ok = len(network_events) == 0
            return {
                "cpu_usage": 45.0,
                "network_connectivity": is_network_ok,
                "network_tamper_events": len(network_events)
            }
        
        self.health_monitor.register_health_check("integration_camera", network_aware_health_check)
        self.health_monitor.run_health_checks()
        
        # Verify health reflects network issues
        component_health = self.health_monitor.get_component_health("integration_camera")
        assert component_health is not None
        assert component_health.metadata["network_connectivity"] is False
        assert component_health.metadata["network_tamper_events"] > 0
    
    def test_comprehensive_monitoring_scenario(self):
        """Test comprehensive monitoring scenario with multiple systems."""
        # Establish tamper detector baseline
        normal_frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.tamper_detector.establish_baseline(normal_frame)
        
        # Create comprehensive health check
        def comprehensive_health_check():
            # Get tamper detection status
            tamper_status = self.tamper_detector.get_detection_status()
            
            # Simulate some tampering
            test_frame = np.random.randint(20, 50, (480, 640, 3), dtype=np.uint8)  # Dark frame
            tamper_events = self.tamper_detector.detect_tampering(test_frame)
            
            # System tampering check
            system_events = self.tamper_detector.detect_system_tampering()
            
            return {
                "cpu_usage": 55.0,
                "memory_usage": 65.0,
                "temperature": 45.0,
                "baseline_established": tamper_status["baseline_established"],
                "tamper_events_detected": len(tamper_events),
                "system_events_detected": len(system_events),
                "last_tamper_check": datetime.now().isoformat(),
                "detection_system_healthy": len(tamper_events) == 0 and len(system_events) == 0
            }
        
        # Register and run health check
        self.health_monitor.register_health_check("integration_camera", comprehensive_health_check)
        self.health_monitor.run_health_checks()
        
        # Verify comprehensive monitoring data
        component_health = self.health_monitor.get_component_health("integration_camera")
        assert component_health is not None
        assert component_health.cpu_usage == 55.0
        assert component_health.memory_usage == 65.0
        assert component_health.temperature == 45.0
        assert "baseline_established" in component_health.metadata
        assert "tamper_events_detected" in component_health.metadata
        assert "system_events_detected" in component_health.metadata
        assert "last_tamper_check" in component_health.metadata
    
    def test_predictive_maintenance_with_tamper_history(self):
        """Test predictive maintenance considering tamper detection history."""
        # Establish baseline
        normal_frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.tamper_detector.establish_baseline(normal_frame)
        
        # Simulate degrading conditions over time
        for i in range(10):
            # Gradually darker frames (simulating lens degradation)
            degraded_frame = np.random.randint(100 - i*8, 200 - i*8, (480, 640, 3), dtype=np.uint8)
            events = self.tamper_detector.detect_tampering(degraded_frame)
            
            # Store health data with tamper event count
            health_data = {
                "cpu_usage": 50.0 + i * 2,  # Increasing CPU usage
                "memory_usage": 60.0 + i * 1.5,
                "tamper_events": len(events),
                "brightness_degradation": i * 8
            }
            
            self.health_monitor._store_health_history("integration_camera", health_data)
            time.sleep(0.01)  # Small delay for timestamp differences
        
        # Run failure prediction
        predictions = self.health_monitor.predict_failures("integration_camera")
        
        # Should have predictions due to increasing trends
        assert "predictions" in predictions or predictions["prediction"] == "insufficient_data"
        
        if "predictions" in predictions:
            # Check if any predictions were made
            assert isinstance(predictions["predictions"], dict)
    
    def test_alert_escalation_integration(self):
        """Test alert escalation when both tamper and health issues occur."""
        # Establish baseline
        normal_frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.tamper_detector.establish_baseline(normal_frame)
        
        # Create critical scenario: tamper + high resource usage
        def critical_health_check():
            # Detect tampering
            critical_frame = np.random.randint(0, 20, (480, 640, 3), dtype=np.uint8)
            tamper_events = self.tamper_detector.detect_tampering(critical_frame)
            
            # Report critical resource usage
            return {
                "cpu_usage": 95.0,  # Critical CPU usage
                "memory_usage": 92.0,  # Critical memory usage
                "temperature": 85.0,  # High temperature
                "tamper_events": len(tamper_events),
                "system_status": "critical" if tamper_events else "warning"
            }
        
        self.health_monitor.register_health_check("integration_camera", critical_health_check)
        self.health_monitor.run_health_checks()
        
        # Verify critical status
        component_health = self.health_monitor.get_component_health("integration_camera")
        assert component_health is not None
        assert component_health.status == ComponentStatus.CRITICAL
        assert component_health.cpu_usage == 95.0
        assert component_health.memory_usage == 92.0
        assert len(component_health.error_messages) > 0
        
        # Check system health reflects critical component
        system_health = self.health_monitor.get_system_health()
        assert system_health.overall_status == ComponentStatus.CRITICAL
        
        # Verify unhealthy components list includes our camera
        unhealthy = self.health_monitor.get_unhealthy_components()
        assert len(unhealthy) > 0
        assert any(comp.component_id == "integration_camera" for comp in unhealthy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])