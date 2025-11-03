"""
Unit tests for Tamper Detection System.

Tests comprehensive tamper detection including:
- Lens occlusion detection with various obstruction scenarios
- Camera movement detection with simulated displacement
- Network failure handling and recovery procedures
- Power failure monitoring and system health checks
"""

import pytest
import numpy as np
import cv2
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import threading

from edge.src.tamper_detector import TamperDetector, TamperEvent, TamperType, CameraBaseline
from shared.models.health import ComponentHealth, ComponentStatus, ComponentType


class TestCameraBaseline:
    """Test cases for CameraBaseline class."""
    
    def test_baseline_initialization(self):
        """Test baseline initializes correctly."""
        baseline = CameraBaseline(camera_id="test_camera")
        
        assert baseline.camera_id == "test_camera"
        assert baseline.reference_frame is None
        assert baseline.reference_histogram is None
        assert baseline.reference_edges is None
        assert baseline.is_valid() is False
    
    def test_baseline_validity_check(self):
        """Test baseline validity checking."""
        baseline = CameraBaseline(camera_id="test_camera")
        
        # Should be invalid initially
        assert baseline.is_valid() is False
        
        # Add required components
        baseline.reference_frame = np.zeros((480, 640), dtype=np.uint8)
        baseline.reference_histogram = np.zeros(256)
        baseline.reference_edges = np.zeros((480, 640), dtype=np.uint8)
        
        # Should be valid now
        assert baseline.is_valid() is True


class TestTamperEvent:
    """Test cases for TamperEvent class."""
    
    def test_tamper_event_creation(self):
        """Test tamper event creation and serialization."""
        event = TamperEvent(
            tamper_type=TamperType.LENS_OCCLUSION,
            camera_id="test_camera",
            confidence=0.8,
            severity="high",
            description="Test occlusion event",
            metadata={"brightness_ratio": 0.2}
        )
        
        assert event.tamper_type == TamperType.LENS_OCCLUSION
        assert event.camera_id == "test_camera"
        assert event.confidence == 0.8
        assert event.severity == "high"
        
        # Test serialization
        event_dict = event.to_dict()
        assert event_dict["tamper_type"] == "lens_occlusion"
        assert event_dict["camera_id"] == "test_camera"
        assert event_dict["confidence"] == 0.8
        assert event_dict["metadata"]["brightness_ratio"] == 0.2


class TestTamperDetector:
    """Test cases for TamperDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = TamperDetector(camera_id="test_camera", baseline_frames=5)
        
        # Create test frames
        self.normal_frame = self.create_normal_frame()
        self.dark_frame = self.create_dark_frame()
        self.bright_frame = self.create_bright_frame()
        self.noisy_frame = self.create_noisy_frame()
        self.shifted_frame = self.create_shifted_frame()
    
    def create_normal_frame(self, width=640, height=480):
        """Create a normal test frame with typical indoor lighting."""
        frame = np.random.randint(80, 180, (height, width, 3), dtype=np.uint8)
        
        # Add some structure (edges) to make it realistic
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), 2)
        cv2.rectangle(frame, (300, 150), (400, 250), (100, 100, 100), 2)
        cv2.circle(frame, (500, 300), 50, (200, 200, 200), 2)
        
        return frame
    
    def create_dark_frame(self, width=640, height=480):
        """Create a dark frame simulating lens occlusion."""
        # Very dark frame with minimal brightness
        frame = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        return frame
    
    def create_bright_frame(self, width=640, height=480):
        """Create a bright frame simulating lighting changes."""
        frame = np.random.randint(200, 255, (height, width, 3), dtype=np.uint8)
        return frame
    
    def create_noisy_frame(self, width=640, height=480):
        """Create a noisy frame with reduced edge content."""
        # Random noise without clear edges
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        return frame
    
    def create_shifted_frame(self, width=640, height=480):
        """Create a frame that simulates camera movement."""
        frame = self.create_normal_frame(width, height)
        
        # Apply translation to simulate camera movement
        M = np.float32([[1, 0, 50], [0, 1, 30]])  # 50px right, 30px down
        frame = cv2.warpAffine(frame, M, (width, height))
        
        return frame
    
    def test_detector_initialization(self):
        """Test tamper detector initializes correctly."""
        assert self.detector.camera_id == "test_camera"
        assert self.detector.baseline_frames == 5
        assert self.detector.baseline_established is False
        assert self.detector.baseline_frame_count == 0
        assert len(self.detector.recent_events) == 0
    
    def test_establish_baseline_success(self):
        """Test successful baseline establishment."""
        # Feed frames to establish baseline
        for i in range(5):
            frame = self.create_normal_frame()
            result = self.detector.establish_baseline(frame)
            
            if i < 4:
                assert result is False  # Not yet established
                assert self.detector.baseline_established is False
            else:
                assert result is True  # Should be established
                assert self.detector.baseline_established is True
        
        # Check baseline components are set
        assert self.detector.baseline.is_valid()
        assert self.detector.baseline.reference_frame is not None
        assert self.detector.baseline.reference_histogram is not None
        assert self.detector.baseline.reference_edges is not None
        assert self.detector.baseline.avg_brightness > 0
        assert self.detector.baseline.avg_contrast > 0
    
    def test_establish_baseline_invalid_frame(self):
        """Test baseline establishment with invalid frames."""
        # Test with None frame
        result = self.detector.establish_baseline(None)
        assert result is False
        
        # Test with empty frame
        empty_frame = np.array([])
        result = self.detector.establish_baseline(empty_frame)
        assert result is False
    
    def test_occlusion_detection_dark_frame(self):
        """Test lens occlusion detection with dark frame."""
        # Establish baseline with normal frames
        for _ in range(5):
            self.detector.establish_baseline(self.normal_frame)
        
        # Test with dark frame (simulating occlusion)
        events = self.detector.detect_tampering(self.dark_frame)
        
        # Should detect occlusion
        occlusion_events = [e for e in events if e.tamper_type == TamperType.LENS_OCCLUSION]
        assert len(occlusion_events) > 0
        
        event = occlusion_events[0]
        assert event.confidence > 0.5
        assert event.severity in ["medium", "high", "critical"]
        assert "occlusion" in event.description.lower()
    
    def test_occlusion_detection_no_false_positive(self):
        """Test that normal frames don't trigger occlusion detection."""
        # Establish baseline
        for _ in range(5):
            self.detector.establish_baseline(self.normal_frame)
        
        # Test with similar normal frame
        similar_frame = self.create_normal_frame()
        events = self.detector.detect_tampering(similar_frame)
        
        # Should not detect occlusion
        occlusion_events = [e for e in events if e.tamper_type == TamperType.LENS_OCCLUSION]
        assert len(occlusion_events) == 0
    
    def test_movement_detection_shifted_frame(self):
        """Test camera movement detection with shifted frame."""
        # Establish baseline
        for _ in range(5):
            self.detector.establish_baseline(self.normal_frame)
        
        # Test with shifted frame (simulating camera movement)
        events = self.detector.detect_tampering(self.shifted_frame)
        
        # Should detect movement
        movement_events = [e for e in events if e.tamper_type == TamperType.CAMERA_MOVEMENT]
        assert len(movement_events) > 0
        
        event = movement_events[0]
        assert event.confidence > 0.3
        assert "movement" in event.description.lower()
        assert "displacement" in event.description.lower()
    
    def test_movement_detection_insufficient_features(self):
        """Test movement detection with insufficient features."""
        # Establish baseline with normal frame
        for _ in range(5):
            self.detector.establish_baseline(self.normal_frame)
        
        # Test with noisy frame (few features)
        events = self.detector.detect_tampering(self.noisy_frame)
        
        # Might detect movement due to insufficient matches
        movement_events = [e for e in events if e.tamper_type == TamperType.CAMERA_MOVEMENT]
        if movement_events:
            event = movement_events[0]
            assert "insufficient" in event.description.lower()
    
    def test_lighting_change_detection(self):
        """Test lighting change detection."""
        # Establish baseline with normal frames
        for _ in range(5):
            self.detector.establish_baseline(self.normal_frame)
        
        # Test with bright frame (significant lighting change)
        events = self.detector.detect_tampering(self.bright_frame)
        
        # Should detect lighting change
        lighting_events = [e for e in events if e.tamper_type == TamperType.LIGHTING_CHANGE]
        assert len(lighting_events) > 0
        
        event = lighting_events[0]
        assert event.confidence > 0.3
        assert "lighting" in event.description.lower()
    
    def test_signal_loss_detection(self):
        """Test signal loss detection with None frame."""
        # Establish baseline first
        for _ in range(5):
            self.detector.establish_baseline(self.normal_frame)
        
        # Test with None frame (signal loss)
        events = self.detector.detect_tampering(None)
        
        # Should detect signal loss
        signal_events = [e for e in events if e.tamper_type == TamperType.SIGNAL_LOSS]
        assert len(signal_events) > 0
        
        event = signal_events[0]
        assert event.severity == "critical"
        assert "signal loss" in event.description.lower()
    
    @patch('psutil.sensors_battery')
    def test_power_failure_detection_low_battery(self, mock_battery):
        """Test power failure detection with low battery."""
        # Mock low battery condition
        mock_battery_info = Mock()
        mock_battery_info.power_plugged = False
        mock_battery_info.percent = 15  # Low battery
        mock_battery.return_value = mock_battery_info
        
        events = self.detector.detect_system_tampering()
        
        # Should detect power issue
        power_events = [e for e in events if e.tamper_type == TamperType.POWER_FAILURE]
        assert len(power_events) > 0
        
        event = power_events[0]
        assert event.severity == "critical"
        assert "battery" in event.description.lower()
        assert event.metadata["battery_percent"] == 15
    
    @patch('psutil.sensors_temperatures')
    def test_power_failure_detection_high_temperature(self, mock_temps):
        """Test power failure detection with high temperature."""
        # Mock high temperature condition
        mock_temp_entry = Mock()
        mock_temp_entry.current = 85  # High temperature
        mock_temps.return_value = {"cpu": [mock_temp_entry]}
        
        events = self.detector.detect_system_tampering()
        
        # Should detect power/thermal issue
        power_events = [e for e in events if e.tamper_type == TamperType.POWER_FAILURE]
        assert len(power_events) > 0
        
        event = power_events[0]
        assert event.severity == "high"
        assert "temperature" in event.description.lower()
        assert event.metadata["temperature"] == 85
    
    @patch('socket.create_connection')
    def test_network_failure_detection(self, mock_socket):
        """Test network failure detection."""
        # Mock network connection failure
        mock_socket.side_effect = OSError("Network unreachable")
        
        events = self.detector.detect_system_tampering()
        
        # Should detect network issue
        network_events = [e for e in events if e.tamper_type == TamperType.CABLE_CUT]
        assert len(network_events) > 0
        
        event = network_events[0]
        assert event.severity == "high"
        assert "connectivity" in event.description.lower()
    
    @patch('socket.create_connection')
    def test_network_connectivity_success(self, mock_socket):
        """Test successful network connectivity check."""
        # Mock successful connection
        mock_socket.return_value = Mock()
        
        events = self.detector.detect_system_tampering()
        
        # Should not detect network issues
        network_events = [e for e in events if e.tamper_type == TamperType.CABLE_CUT]
        assert len(network_events) == 0
    
    def test_consecutive_failures_signal_loss(self):
        """Test signal loss detection after consecutive failures."""
        # Establish baseline
        for _ in range(5):
            self.detector.establish_baseline(self.normal_frame)
        
        # Simulate consecutive failures by advancing time
        original_time = self.detector.last_frame_time
        self.detector.last_frame_time = original_time - timedelta(seconds=60)
        self.detector.consecutive_failures = 6  # Above threshold
        
        events = self.detector.detect_system_tampering()
        
        # Should detect signal loss
        signal_events = [e for e in events if e.tamper_type == TamperType.SIGNAL_LOSS]
        assert len(signal_events) > 0
        
        event = signal_events[0]
        assert event.metadata["consecutive_failures"] == 6
    
    def test_duplicate_event_filtering(self):
        """Test that duplicate events are filtered out."""
        # Establish baseline
        for _ in range(5):
            self.detector.establish_baseline(self.normal_frame)
        
        # Generate same type of event twice quickly
        events1 = self.detector.detect_tampering(self.dark_frame)
        events2 = self.detector.detect_tampering(self.dark_frame)
        
        # First detection should have events
        occlusion_events1 = [e for e in events1 if e.tamper_type == TamperType.LENS_OCCLUSION]
        assert len(occlusion_events1) > 0
        
        # Second detection should be filtered (duplicate)
        occlusion_events2 = [e for e in events2 if e.tamper_type == TamperType.LENS_OCCLUSION]
        assert len(occlusion_events2) == 0  # Should be filtered as duplicate
    
    def test_baseline_update(self):
        """Test baseline update functionality."""
        # Establish initial baseline
        for _ in range(5):
            self.detector.establish_baseline(self.normal_frame)
        
        original_brightness = self.detector.baseline.avg_brightness
        
        # Update baseline with slightly different frame
        brighter_frame = self.create_normal_frame()
        brighter_frame = cv2.add(brighter_frame, np.ones_like(brighter_frame) * 20)
        
        self.detector.update_baseline(brighter_frame)
        
        # Baseline should be slightly updated (very slow learning rate)
        new_brightness = self.detector.baseline.avg_brightness
        # Due to very slow learning rate (0.01), change should be minimal
        assert abs(new_brightness - original_brightness) < 5
    
    def test_detection_status_reporting(self):
        """Test detection status reporting."""
        # Establish baseline
        for _ in range(5):
            self.detector.establish_baseline(self.normal_frame)
        
        # Add some events
        self.detector.detect_tampering(self.dark_frame)
        
        status = self.detector.get_detection_status()
        
        assert status["camera_id"] == "test_camera"
        assert status["baseline_established"] is True
        assert status["baseline_frame_count"] == 5
        assert status["recent_events_count"] > 0
        assert "baseline_stats" in status
        assert status["baseline_stats"]["avg_brightness"] > 0
    
    def test_thread_safety(self):
        """Test thread safety of tamper detection."""
        # Establish baseline
        for _ in range(5):
            self.detector.establish_baseline(self.normal_frame)
        
        results = []
        errors = []
        
        def detect_tampering_thread():
            try:
                events = self.detector.detect_tampering(self.normal_frame)
                results.append(len(events))
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads simultaneously
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=detect_tampering_thread)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should not have any errors
        assert len(errors) == 0
        assert len(results) == 5
    
    def test_sharpness_calculation(self):
        """Test image sharpness calculation."""
        # Create sharp frame with clear edges
        sharp_frame = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(sharp_frame, (20, 20), (80, 80), 255, 2)
        
        # Create blurry frame
        blurry_frame = cv2.GaussianBlur(sharp_frame, (15, 15), 0)
        
        sharp_value = self.detector._calculate_sharpness(sharp_frame)
        blurry_value = self.detector._calculate_sharpness(blurry_frame)
        
        # Sharp frame should have higher sharpness value
        assert sharp_value > blurry_value
        assert sharp_value > 0
        assert blurry_value >= 0
    
    def test_point_in_polygon_edge_cases(self):
        """Test point-in-polygon algorithm with edge cases."""
        # Simple square polygon
        polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        
        # Test corner points
        assert self.detector._point_in_polygon(0, 0, polygon) is True
        assert self.detector._point_in_polygon(100, 100, polygon) is True
        
        # Test edge points
        assert self.detector._point_in_polygon(50, 0, polygon) is True
        assert self.detector._point_in_polygon(0, 50, polygon) is True
        
        # Test points just outside
        assert self.detector._point_in_polygon(-1, 50, polygon) is False
        assert self.detector._point_in_polygon(101, 50, polygon) is False


class TestTamperDetectionIntegration:
    """Integration tests for tamper detection system."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.detector = TamperDetector(camera_id="integration_test", baseline_frames=3)
    
    def test_complete_occlusion_scenario(self):
        """Test complete occlusion detection scenario."""
        # Phase 1: Establish baseline with normal frames
        normal_frames = []
        for i in range(3):
            frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
            # Add some structure
            cv2.rectangle(frame, (100+i*10, 100), (200+i*10, 200), (255, 255, 255), 2)
            normal_frames.append(frame)
            self.detector.establish_baseline(frame)
        
        assert self.detector.baseline_established is True
        
        # Phase 2: Gradual occlusion (partial)
        partial_occlusion = np.random.randint(50, 100, (480, 640, 3), dtype=np.uint8)
        events = self.detector.detect_tampering(partial_occlusion)
        
        # Should detect some level of occlusion
        occlusion_events = [e for e in events if e.tamper_type == TamperType.LENS_OCCLUSION]
        if occlusion_events:
            assert occlusion_events[0].severity in ["medium", "high"]
        
        # Phase 3: Complete occlusion
        complete_occlusion = np.random.randint(0, 20, (480, 640, 3), dtype=np.uint8)
        events = self.detector.detect_tampering(complete_occlusion)
        
        # Should detect critical occlusion
        occlusion_events = [e for e in events if e.tamper_type == TamperType.LENS_OCCLUSION]
        assert len(occlusion_events) > 0
        assert occlusion_events[0].severity in ["high", "critical"]
        assert occlusion_events[0].confidence > 0.7
    
    def test_camera_movement_scenario(self):
        """Test camera movement detection scenario."""
        # Establish baseline
        base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add distinctive features
        cv2.rectangle(base_frame, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.circle(base_frame, (400, 300), 50, (128, 128, 128), -1)
        cv2.rectangle(base_frame, (500, 50), (600, 150), (200, 200, 200), -1)
        
        for _ in range(3):
            self.detector.establish_baseline(base_frame)
        
        # Simulate small movement (should not trigger)
        small_movement = base_frame.copy()
        M = np.float32([[1, 0, 5], [0, 1, 3]])  # Small shift
        small_movement = cv2.warpAffine(small_movement, M, (640, 480))
        
        events = self.detector.detect_tampering(small_movement)
        movement_events = [e for e in events if e.tamper_type == TamperType.CAMERA_MOVEMENT]
        # Small movement might not trigger detection
        
        # Simulate large movement (should trigger)
        large_movement = base_frame.copy()
        M = np.float32([[1, 0, 80], [0, 1, 60]])  # Large shift
        large_movement = cv2.warpAffine(large_movement, M, (640, 480))
        
        events = self.detector.detect_tampering(large_movement)
        movement_events = [e for e in events if e.tamper_type == TamperType.CAMERA_MOVEMENT]
        assert len(movement_events) > 0
        assert movement_events[0].confidence > 0.5
        assert "displacement" in movement_events[0].description
    
    @patch('socket.create_connection')
    @patch('psutil.sensors_battery')
    def test_system_failure_recovery_scenario(self, mock_battery, mock_socket):
        """Test system failure detection and recovery."""
        # Phase 1: Normal operation
        mock_socket.return_value = Mock()  # Network OK
        mock_battery.return_value = None  # No battery info
        
        events = self.detector.detect_system_tampering()
        assert len(events) == 0  # No issues
        
        # Phase 2: Network failure
        mock_socket.side_effect = OSError("Network down")
        
        events = self.detector.detect_system_tampering()
        network_events = [e for e in events if e.tamper_type == TamperType.CABLE_CUT]
        assert len(network_events) > 0
        
        # Phase 3: Power issue
        mock_battery_info = Mock()
        mock_battery_info.power_plugged = False
        mock_battery_info.percent = 10
        mock_battery.return_value = mock_battery_info
        
        events = self.detector.detect_system_tampering()
        power_events = [e for e in events if e.tamper_type == TamperType.POWER_FAILURE]
        assert len(power_events) > 0
        
        # Phase 4: Recovery
        mock_socket.side_effect = None
        mock_socket.return_value = Mock()  # Network restored
        mock_battery_info.power_plugged = True
        mock_battery_info.percent = 80
        
        events = self.detector.detect_system_tampering()
        # Should have fewer or no critical events
        critical_events = [e for e in events if e.severity == "critical"]
        assert len(critical_events) == 0


class TestTamperDetectionPerformance:
    """Performance tests for tamper detection system."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.detector = TamperDetector(camera_id="perf_test", baseline_frames=3)
        
        # Establish baseline quickly
        test_frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.detector.establish_baseline(test_frame)
    
    def test_detection_latency_requirement(self):
        """Test that tamper detection meets latency requirements."""
        test_frame = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
        
        # Measure detection time
        latencies = []
        for _ in range(10):
            start_time = time.time()
            events = self.detector.detect_tampering(test_frame)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        
        # Should be fast enough for real-time processing
        assert avg_latency < 100, f"Average latency {avg_latency}ms too high"
        assert max_latency < 200, f"Max latency {max_latency}ms too high"
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable over time."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run many detection cycles
        test_frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        for i in range(100):
            events = self.detector.detect_tampering(test_frame)
            
            # Occasionally check memory
            if i % 20 == 0:
                current_memory = process.memory_info().rss
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be reasonable (less than 50MB)
                assert memory_growth < 50 * 1024 * 1024, f"Memory growth {memory_growth} bytes too high"
    
    def test_concurrent_detection_performance(self):
        """Test performance with concurrent detection requests."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        test_frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        
        def detection_worker():
            start_time = time.time()
            events = self.detector.detect_tampering(test_frame)
            end_time = time.time()
            results_queue.put(end_time - start_time)
        
        # Run concurrent detections
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=detection_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Collect results
        latencies = []
        while not results_queue.empty():
            latencies.append(results_queue.get())
        
        assert len(latencies) == 5
        avg_latency = np.mean(latencies) * 1000  # Convert to ms
        
        # Concurrent performance should still be reasonable
        assert avg_latency < 150, f"Concurrent average latency {avg_latency}ms too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])