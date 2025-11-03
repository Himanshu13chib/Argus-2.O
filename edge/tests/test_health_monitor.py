"""
Unit tests for Health Monitor System.

Tests comprehensive health monitoring including:
- Component registration and health tracking
- Network connectivity monitoring and offline mode handling
- Predictive maintenance alerts and system diagnostics
- Performance monitoring and trend analysis
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from edge.src.health_monitor import HealthMonitor, HealthCheckResult, PredictiveAlert
from shared.models.health import HealthStatus, ComponentHealth, ComponentType, ComponentStatus


class TestHealthCheckResult:
    """Test cases for HealthCheckResult class."""
    
    def test_health_check_result_creation(self):
        """Test health check result creation."""
        result = HealthCheckResult(
            component_id="test_component",
            success=True,
            metrics={"cpu_usage": 45.2, "memory_usage": 60.1},
            errors=["Error 1"],
            warnings=["Warning 1"]
        )
        
        assert result.component_id == "test_component"
        assert result.success is True
        assert result.metrics["cpu_usage"] == 45.2
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        assert isinstance(result.timestamp, datetime)


class TestPredictiveAlert:
    """Test cases for PredictiveAlert class."""
    
    def test_predictive_alert_creation(self):
        """Test predictive alert creation."""
        future_time = datetime.now() + timedelta(hours=2)
        alert = PredictiveAlert(
            component_id="test_component",
            alert_type="cpu_overload",
            predicted_failure_time=future_time,
            confidence=0.85,
            description="CPU usage trending upward",
            recommended_actions=["Scale resources", "Check processes"]
        )
        
        assert alert.component_id == "test_component"
        assert alert.alert_type == "cpu_overload"
        assert alert.confidence == 0.85
        assert len(alert.recommended_actions) == 2


class TestHealthMonitor:
    """Test cases for HealthMonitor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = HealthMonitor(check_interval=1, history_retention_hours=1)
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.monitor.is_running:
            self.monitor.stop_monitoring()
    
    def test_monitor_initialization(self):
        """Test health monitor initializes correctly."""
        assert self.monitor.check_interval == 1
        assert self.monitor.history_retention_hours == 1
        assert len(self.monitor.components) == 0
        assert self.monitor.is_running is False
        assert self.monitor.is_offline_mode is False
        
        # Check default thresholds are set
        assert ComponentType.CAMERA in self.monitor.alert_thresholds
        assert ComponentType.EDGE_NODE in self.monitor.alert_thresholds
        assert self.monitor.alert_thresholds[ComponentType.CAMERA]["cpu_usage"] == 80.0
    
    def test_register_component(self):
        """Test component registration."""
        self.monitor.register_component(
            component_id="test_camera",
            component_type=ComponentType.CAMERA,
            check_interval_seconds=30
        )
        
        assert "test_camera" in self.monitor.components
        assert self.monitor.components["test_camera"].component_type == ComponentType.CAMERA
        assert self.monitor.component_intervals["test_camera"] == 30
        assert "test_camera" in self.monitor.system_health.components
    
    def test_unregister_component(self):
        """Test component unregistration."""
        # Register first
        self.monitor.register_component("test_camera", ComponentType.CAMERA)
        assert "test_camera" in self.monitor.components
        
        # Unregister
        self.monitor.unregister_component("test_camera")
        assert "test_camera" not in self.monitor.components
        assert "test_camera" not in self.monitor.component_intervals
        assert "test_camera" not in self.monitor.system_health.components
    
    def test_update_component_health(self):
        """Test component health updates."""
        # Register component
        self.monitor.register_component("test_camera", ComponentType.CAMERA)
        
        # Update health data
        health_data = {
            "cpu_usage": 45.5,
            "memory_usage": 60.2,
            "temperature": 65.0,
            "uptime_seconds": 3600,
            "metadata": {"location": "entrance"}
        }
        
        self.monitor.update_component_health("test_camera", health_data)
        
        component = self.monitor.components["test_camera"]
        assert component.cpu_usage == 45.5
        assert component.memory_usage == 60.2
        assert component.temperature == 65.0
        assert component.uptime_seconds == 3600
        assert component.metadata["location"] == "entrance"
        assert component.status == ComponentStatus.HEALTHY
    
    def test_component_health_threshold_warning(self):
        """Test component health threshold warnings."""
        # Register component
        self.monitor.register_component("test_camera", ComponentType.CAMERA)
        
        # Update with high CPU usage (above warning threshold)
        health_data = {"cpu_usage": 85.0}  # Above 80% threshold
        self.monitor.update_component_health("test_camera", health_data)
        
        component = self.monitor.components["test_camera"]
        assert component.status == ComponentStatus.WARNING
        assert len(component.warning_messages) > 0
        assert "cpu usage" in component.warning_messages[0].lower()
    
    def test_component_health_threshold_critical(self):
        """Test component health threshold critical alerts."""
        # Register component
        self.monitor.register_component("test_camera", ComponentType.CAMERA)
        
        # Update with critical CPU usage
        health_data = {"cpu_usage": 95.0}  # Well above threshold
        self.monitor.update_component_health("test_camera", health_data)
        
        component = self.monitor.components["test_camera"]
        assert component.status == ComponentStatus.CRITICAL
        assert len(component.error_messages) > 0
        assert "critical cpu usage" in component.error_messages[0].lower()
    
    def test_get_component_health(self):
        """Test getting component health."""
        # Register and update component
        self.monitor.register_component("test_camera", ComponentType.CAMERA)
        health_data = {"cpu_usage": 50.0}
        self.monitor.update_component_health("test_camera", health_data)
        
        # Get health
        health = self.monitor.get_component_health("test_camera")
        assert health is not None
        assert health.component_id == "test_camera"
        assert health.cpu_usage == 50.0
        
        # Test non-existent component
        health = self.monitor.get_component_health("non_existent")
        assert health is None
    
    def test_get_unhealthy_components(self):
        """Test getting unhealthy components."""
        # Register components
        self.monitor.register_component("healthy_camera", ComponentType.CAMERA)
        self.monitor.register_component("unhealthy_camera", ComponentType.CAMERA)
        
        # Update with different health states
        self.monitor.update_component_health("healthy_camera", {"cpu_usage": 50.0})
        self.monitor.update_component_health("unhealthy_camera", {"cpu_usage": 95.0})
        
        unhealthy = self.monitor.get_unhealthy_components()
        assert len(unhealthy) == 1
        assert unhealthy[0].component_id == "unhealthy_camera"
    
    def test_register_health_check(self):
        """Test registering custom health check functions."""
        def custom_health_check():
            return {"custom_metric": 42.0, "status": "ok"}
        
        self.monitor.register_health_check("test_component", custom_health_check)
        assert "test_component" in self.monitor.health_checks
    
    def test_run_health_checks(self):
        """Test running registered health checks."""
        # Register component and health check
        self.monitor.register_component("test_component", ComponentType.EDGE_NODE)
        
        def mock_health_check():
            return {"cpu_usage": 30.0, "memory_usage": 40.0}
        
        self.monitor.register_health_check("test_component", mock_health_check)
        
        # Run health checks
        self.monitor.run_health_checks()
        
        # Verify component was updated
        component = self.monitor.components["test_component"]
        assert component.cpu_usage == 30.0
        assert component.memory_usage == 40.0
    
    def test_run_health_checks_with_failure(self):
        """Test health check handling when check function fails."""
        # Register component and failing health check
        self.monitor.register_component("test_component", ComponentType.EDGE_NODE)
        
        def failing_health_check():
            raise Exception("Health check failed")
        
        self.monitor.register_health_check("test_component", failing_health_check)
        
        # Run health checks (should not raise exception)
        self.monitor.run_health_checks()
        
        # Component should have error recorded
        component = self.monitor.components["test_component"]
        assert component.status == ComponentStatus.CRITICAL
        assert len(component.error_messages) > 0
    
    def test_set_alert_thresholds(self):
        """Test setting custom alert thresholds."""
        custom_thresholds = {
            "cpu_usage": 70.0,
            "memory_usage": 80.0,
            "temperature": 60.0
        }
        
        self.monitor.set_alert_thresholds(ComponentType.CAMERA, custom_thresholds)
        
        assert self.monitor.alert_thresholds[ComponentType.CAMERA]["cpu_usage"] == 70.0
        assert self.monitor.alert_thresholds[ComponentType.CAMERA]["memory_usage"] == 80.0
        assert self.monitor.alert_thresholds[ComponentType.CAMERA]["temperature"] == 60.0
    
    def test_get_health_history(self):
        """Test getting component health history."""
        # Register component
        self.monitor.register_component("test_component", ComponentType.CAMERA)
        
        # Add some history manually
        for i in range(5):
            health_data = {"cpu_usage": 50.0 + i * 5}
            self.monitor._store_health_history("test_component", health_data)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Get history
        history = self.monitor.get_health_history("test_component", hours=1)
        assert len(history) == 5
        
        # Check data is in chronological order
        for i in range(1, len(history)):
            assert history[i]["timestamp"] >= history[i-1]["timestamp"]
    
    def test_get_system_metrics(self):
        """Test getting system-wide metrics."""
        # Register some components
        self.monitor.register_component("camera1", ComponentType.CAMERA)
        self.monitor.register_component("camera2", ComponentType.CAMERA)
        self.monitor.register_component("edge1", ComponentType.EDGE_NODE)
        
        # Update with health data
        self.monitor.update_component_health("camera1", {"cpu_usage": 50.0})
        self.monitor.update_component_health("camera2", {"cpu_usage": 60.0})
        self.monitor.update_component_health("edge1", {"cpu_usage": 70.0})
        
        metrics = self.monitor.get_system_metrics()
        
        assert metrics["total_components"] == 3
        assert metrics["healthy_components"] == 3
        assert metrics["system_availability"] == 100.0
        assert metrics["avg_cpu_usage"] == 60.0  # (50+60+70)/3
        assert "system_uptime_hours" in metrics
        assert "last_update" in metrics
    
    @patch('socket.create_connection')
    def test_network_connectivity_check_success(self, mock_socket):
        """Test successful network connectivity check."""
        mock_socket.return_value = Mock()
        
        is_connected = self.monitor._check_network_connectivity()
        assert is_connected is True
    
    @patch('socket.create_connection')
    def test_network_connectivity_check_failure(self, mock_socket):
        """Test failed network connectivity check."""
        mock_socket.side_effect = OSError("Network unreachable")
        
        is_connected = self.monitor._check_network_connectivity()
        assert is_connected is False
    
    @patch('socket.create_connection')
    def test_offline_mode_handling(self, mock_socket):
        """Test offline mode activation and deactivation."""
        # Initially online
        mock_socket.return_value = Mock()
        assert self.monitor.is_offline_mode is False
        
        # Simulate network failure
        mock_socket.side_effect = OSError("Network down")
        self.monitor._check_and_handle_network_connectivity()
        assert self.monitor.is_offline_mode is True
        
        # Simulate network recovery
        mock_socket.side_effect = None
        mock_socket.return_value = Mock()
        self.monitor._check_and_handle_network_connectivity()
        assert self.monitor.is_offline_mode is False
    
    def test_enable_disable_offline_mode(self):
        """Test manual offline mode control."""
        assert self.monitor.is_offline_mode is False
        
        self.monitor.enable_offline_mode()
        assert self.monitor.is_offline_mode is True
        
        self.monitor.disable_offline_mode()
        assert self.monitor.is_offline_mode is False
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    @patch('psutil.pids')
    def test_system_health_checks(self, mock_pids, mock_net, mock_disk, mock_memory, mock_cpu):
        """Test system-level health checks."""
        # Mock system metrics
        mock_cpu.return_value = 45.5
        
        mock_memory_info = Mock()
        mock_memory_info.percent = 60.2
        mock_memory.return_value = mock_memory_info
        
        mock_disk_info = Mock()
        mock_disk_info.percent = 70.1
        mock_disk.return_value = mock_disk_info
        
        mock_net_info = Mock()
        mock_net_info.bytes_sent = 1000000
        mock_net_info.bytes_recv = 2000000
        mock_net.return_value = mock_net_info
        
        mock_pids.return_value = [1, 2, 3, 4, 5]  # 5 processes
        
        # Run system health checks
        self.monitor._run_system_health_checks()
        
        # Check that system metrics were stored
        assert "system" in self.monitor.health_history
        history = list(self.monitor.health_history["system"])
        assert len(history) > 0
        
        latest_entry = history[-1]
        assert latest_entry["data"]["cpu_usage"] == 45.5
        assert latest_entry["data"]["memory_usage"] == 60.2
        assert latest_entry["data"]["disk_usage"] == 70.1
        assert latest_entry["data"]["process_count"] == 5
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping the monitoring system."""
        assert self.monitor.is_running is False
        
        # Start monitoring
        self.monitor.start_monitoring()
        assert self.monitor.is_running is True
        assert self.monitor.monitor_thread is not None
        
        # Give it a moment to start
        time.sleep(0.1)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        assert self.monitor.is_running is False
    
    def test_start_monitoring_already_running(self):
        """Test starting monitoring when already running."""
        self.monitor.start_monitoring()
        assert self.monitor.is_running is True
        
        # Try to start again (should not create new thread)
        original_thread = self.monitor.monitor_thread
        self.monitor.start_monitoring()
        assert self.monitor.monitor_thread == original_thread
        
        self.monitor.stop_monitoring()
    
    def test_cleanup_old_history(self):
        """Test cleanup of old health history."""
        # Add some old history
        old_time = datetime.now() - timedelta(hours=25)  # Older than retention
        recent_time = datetime.now() - timedelta(minutes=30)  # Within retention
        
        self.monitor.health_history["test_component"].append({
            "timestamp": old_time,
            "data": {"cpu_usage": 50.0}
        })
        self.monitor.health_history["test_component"].append({
            "timestamp": recent_time,
            "data": {"cpu_usage": 60.0}
        })
        
        # Run cleanup
        self.monitor._cleanup_old_history()
        
        # Old entry should be removed, recent should remain
        history = list(self.monitor.health_history["test_component"])
        assert len(history) == 1
        assert history[0]["timestamp"] == recent_time
    
    def test_analyze_metric_trend_increasing(self):
        """Test metric trend analysis for increasing trend."""
        # Create history with increasing values
        history = []
        for i in range(10):
            history.append({
                "timestamp": datetime.now() - timedelta(minutes=10-i),
                "data": {"cpu_usage": 50.0 + i * 2.0}  # Increasing trend
            })
        
        trend = self.monitor._analyze_metric_trend(history, "cpu_usage")
        
        assert trend["trend"] == "increasing"
        assert trend["rate"] > 0
        assert trend["confidence"] > 0.5
        assert trend["data_points"] == 10
    
    def test_analyze_metric_trend_stable(self):
        """Test metric trend analysis for stable trend."""
        # Create history with stable values
        history = []
        for i in range(10):
            history.append({
                "timestamp": datetime.now() - timedelta(minutes=10-i),
                "data": {"cpu_usage": 50.0 + (i % 2) * 0.5}  # Stable with small variation
            })
        
        trend = self.monitor._analyze_metric_trend(history, "cpu_usage")
        
        assert trend["trend"] == "stable"
        assert abs(trend["rate"]) < 0.5
    
    def test_analyze_metric_trend_insufficient_data(self):
        """Test metric trend analysis with insufficient data."""
        # Create history with too few points
        history = [
            {"timestamp": datetime.now(), "data": {"cpu_usage": 50.0}},
            {"timestamp": datetime.now(), "data": {"cpu_usage": 51.0}}
        ]
        
        trend = self.monitor._analyze_metric_trend(history, "cpu_usage")
        
        assert trend["trend"] == "unknown"
        assert trend["confidence"] == 0.0
    
    def test_predict_failures_insufficient_data(self):
        """Test failure prediction with insufficient data."""
        predictions = self.monitor.predict_failures("non_existent_component")
        assert predictions["prediction"] == "insufficient_data"
    
    def test_predict_failures_with_trends(self):
        """Test failure prediction with trend data."""
        component_id = "test_component"
        
        # Add history with increasing CPU usage
        for i in range(15):
            self.monitor.health_history[component_id].append({
                "timestamp": datetime.now() - timedelta(minutes=15-i),
                "data": {
                    "cpu_usage": 60.0 + i * 2.0,  # Increasing trend
                    "memory_usage": 50.0 + i * 1.5,
                    "temperature": 40.0 + i * 1.0,
                    "error_count": i // 3  # Slowly increasing errors
                }
            })
        
        predictions = self.monitor.predict_failures(component_id)
        
        assert "predictions" in predictions
        assert predictions["component_id"] == component_id
        assert predictions["data_points_analyzed"] == 15
        
        # Should have some predictions due to increasing trends
        if "cpu_overload" in predictions["predictions"]:
            cpu_pred = predictions["predictions"]["cpu_overload"]
            assert cpu_pred["risk"] in ["medium", "high"]
            assert cpu_pred["confidence"] > 0.0
    
    def test_error_trend_analysis(self):
        """Test error rate trend analysis."""
        # Create history with increasing error counts
        history = []
        for i in range(20):
            history.append({
                "timestamp": datetime.now() - timedelta(minutes=20-i),
                "data": {"error_count": i // 4}  # Gradually increasing errors
            })
        
        trend = self.monitor._analyze_error_trend(history)
        
        assert trend["trend"] in ["increasing", "stable"]
        assert trend["confidence"] >= 0.0
    
    def test_estimate_failure_time(self):
        """Test failure time estimation."""
        trend_data = {
            "trend": "increasing",
            "rate": 2.0,  # 2 units per hour
            "confidence": 0.8
        }
        
        # Test with reasonable failure threshold
        failure_time = self.monitor._estimate_failure_time(trend_data, 90.0)
        
        # Should return a time estimate
        assert failure_time is not None
        assert isinstance(failure_time, str)
    
    def test_estimate_failure_time_no_trend(self):
        """Test failure time estimation with no increasing trend."""
        trend_data = {
            "trend": "stable",
            "rate": 0.1,
            "confidence": 0.5
        }
        
        failure_time = self.monitor._estimate_failure_time(trend_data, 90.0)
        assert failure_time is None
    
    def test_update_system_metrics(self):
        """Test system metrics update."""
        # Register components
        self.monitor.register_component("camera1", ComponentType.CAMERA)
        self.monitor.register_component("camera2", ComponentType.CAMERA)
        self.monitor.register_component("edge1", ComponentType.EDGE_NODE)
        
        # Update health (one unhealthy)
        self.monitor.update_component_health("camera1", {"cpu_usage": 50.0})
        self.monitor.update_component_health("camera2", {"cpu_usage": 95.0})  # Unhealthy
        self.monitor.update_component_health("edge1", {"cpu_usage": 60.0})
        
        # Update system metrics
        self.monitor._update_system_metrics()
        
        health = self.monitor.system_health
        assert health.total_cameras == 2
        assert health.active_cameras == 1  # Only one healthy camera
        assert health.total_edge_nodes == 1
        assert health.active_edge_nodes == 1
    
    def test_thread_safety(self):
        """Test thread safety of health monitoring operations."""
        # Register component
        self.monitor.register_component("test_component", ComponentType.CAMERA)
        
        results = []
        errors = []
        
        def update_health_worker():
            try:
                for i in range(10):
                    health_data = {"cpu_usage": 50.0 + i}
                    self.monitor.update_component_health("test_component", health_data)
                    time.sleep(0.001)
                results.append("success")
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=update_health_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should not have errors
        assert len(errors) == 0
        assert len(results) == 3
        
        # Component should still be valid
        component = self.monitor.get_component_health("test_component")
        assert component is not None
        assert component.cpu_usage is not None


class TestHealthMonitorIntegration:
    """Integration tests for health monitoring system."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.monitor = HealthMonitor(check_interval=0.1, history_retention_hours=1)
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.monitor.is_running:
            self.monitor.stop_monitoring()
    
    def test_complete_monitoring_cycle(self):
        """Test complete monitoring cycle with real components."""
        # Register components
        self.monitor.register_component("camera1", ComponentType.CAMERA)
        self.monitor.register_component("edge1", ComponentType.EDGE_NODE)
        
        # Register health checks
        def camera_health_check():
            return {
                "cpu_usage": 45.0,
                "memory_usage": 55.0,
                "temperature": 60.0
            }
        
        def edge_health_check():
            return {
                "cpu_usage": 70.0,
                "memory_usage": 80.0,
                "disk_usage": 65.0
            }
        
        self.monitor.register_health_check("camera1", camera_health_check)
        self.monitor.register_health_check("edge1", edge_health_check)
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Let it run for a short time
        time.sleep(0.5)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Check that components were updated
        camera_health = self.monitor.get_component_health("camera1")
        edge_health = self.monitor.get_component_health("edge1")
        
        assert camera_health is not None
        assert camera_health.cpu_usage == 45.0
        assert edge_health is not None
        assert edge_health.cpu_usage == 70.0
        
        # Check system health
        system_health = self.monitor.get_system_health()
        assert system_health.total_cameras == 1
        assert system_health.total_edge_nodes == 1
    
    @patch('socket.create_connection')
    def test_network_failure_recovery_cycle(self, mock_socket):
        """Test network failure detection and recovery."""
        # Start with good network
        mock_socket.return_value = Mock()
        self.monitor.start_monitoring()
        
        # Let it run briefly
        time.sleep(0.2)
        assert self.monitor.is_offline_mode is False
        
        # Simulate network failure
        mock_socket.side_effect = OSError("Network down")
        time.sleep(0.2)  # Let monitoring cycle detect failure
        
        # Should be in offline mode
        assert self.monitor.is_offline_mode is True
        
        # Simulate network recovery
        mock_socket.side_effect = None
        mock_socket.return_value = Mock()
        time.sleep(0.2)  # Let monitoring cycle detect recovery
        
        # Should be back online
        assert self.monitor.is_offline_mode is False
        
        self.monitor.stop_monitoring()
    
    def test_predictive_alerts_generation(self):
        """Test predictive alerts generation in monitoring loop."""
        # Register component
        self.monitor.register_component("test_component", ComponentType.CAMERA)
        
        # Add trending data manually
        for i in range(20):
            self.monitor.health_history["test_component"].append({
                "timestamp": datetime.now() - timedelta(minutes=20-i),
                "data": {
                    "cpu_usage": 50.0 + i * 2.0,  # Strong upward trend
                    "memory_usage": 40.0 + i * 1.5,
                    "error_count": i // 5
                }
            })
        
        # Start monitoring (will generate predictive alerts)
        self.monitor.start_monitoring()
        time.sleep(0.3)  # Let it run a few cycles
        self.monitor.stop_monitoring()
        
        # Check that predictions were generated (logged)
        # This is mainly testing that the prediction code runs without errors
        predictions = self.monitor.predict_failures("test_component")
        assert "predictions" in predictions or predictions["prediction"] == "analysis_error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])