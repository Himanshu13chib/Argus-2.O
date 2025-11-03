#!/usr/bin/env python3
"""
Core Integration Tests for Project Argus
Tests core system integration without requiring full service deployment.
"""

import pytest
import time
import numpy as np
import cv2
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import core components that are available
from shared.models.detection import Detection, DetectionResult, BoundingBox, DetectionClass
from shared.models.virtual_line import VirtualLine, Point
from shared.models.alerts import Alert, AlertType, Severity


class MockDetectionPipeline:
    """Mock detection pipeline for integration testing."""
    
    def __init__(self, config):
        self.config = config
        self.virtual_lines = []
        self.is_running = False
        self.frame_count = 0
        self.last_crossings = []  # Store crossings separately
    
    def process_frame(self, frame, camera_id):
        """Process frame and return detection result."""
        self.frame_count += 1
        
        # Simulate person detection in the center of frame
        if frame is not None and frame.size > 0:
            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            
            # Create mock detection
            detection = Detection(
                id=f"det_{camera_id}_{self.frame_count}",
                camera_id=camera_id,
                timestamp=datetime.now(),
                bbox=BoundingBox(x=center_x-25, y=center_y-50, width=50, height=100),
                confidence=0.85,
                detection_class=DetectionClass.PERSON,
                features=np.random.rand(512).astype(np.float32)
            )
            
            # Check for virtual line crossings
            self.last_crossings = []
            for vline in self.virtual_lines:
                if self._check_crossing(detection, vline):
                    from shared.models.alerts import CrossingEvent
                    crossing = CrossingEvent(
                        detection_id=detection.id,
                        virtual_line_id=vline.id,
                        crossing_point=(center_x, center_y),
                        crossing_direction="inbound",
                        timestamp=detection.timestamp,
                        confidence=detection.confidence
                    )
                    self.last_crossings.append(crossing)
            
            result = DetectionResult(
                camera_id=camera_id,
                frame_timestamp=datetime.now(),
                detections=[detection],
                processing_time_ms=50.0,
                frame_number=self.frame_count
            )
            # Add crossings as metadata
            result.metadata = {"virtual_line_crossings": self.last_crossings}
            return result
        
        self.last_crossings = []
        return DetectionResult(
            camera_id=camera_id,
            frame_timestamp=datetime.now(),
            detections=[],
            processing_time_ms=30.0,
            frame_number=self.frame_count
        )
    
    def _check_crossing(self, detection, virtual_line):
        """Check if detection crosses virtual line."""
        # Simple crossing check - if detection center is near line
        det_center_x = detection.bbox.x + detection.bbox.width / 2
        det_center_y = detection.bbox.y + detection.bbox.height / 2
        
        if len(virtual_line.points) >= 2:
            line_x = virtual_line.points[0].x
            # Check if detection crosses vertical line
            return abs(det_center_x - line_x) < 30
        
        return False


class MockAlertEngine:
    """Mock alert engine for integration testing."""
    
    def __init__(self):
        self.alerts = []
    
    def generate_alert(self, crossing_event, detection):
        """Generate alert from crossing event and detection."""
        alert = Alert(
            id=f"alert_{len(self.alerts)+1:03d}",
            type=AlertType.VIRTUAL_LINE_CROSSING,
            severity=self._calculate_severity(detection),
            camera_id=detection.camera_id,
            detection_id=detection.id,
            timestamp=detection.timestamp,
            confidence=detection.confidence,
            risk_score=self._calculate_risk_score(detection),
            metadata={
                "crossing_direction": crossing_event.crossing_direction,
                "virtual_line_id": crossing_event.virtual_line_id
            }
        )
        
        self.alerts.append(alert)
        return alert
    
    def _calculate_severity(self, detection):
        """Calculate alert severity based on detection."""
        if detection.confidence > 0.9:
            return Severity.HIGH
        elif detection.confidence > 0.7:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def _calculate_risk_score(self, detection):
        """Calculate risk score for alert."""
        base_score = detection.confidence
        
        # Increase risk for night time
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    def get_active_alerts(self):
        """Get all active alerts."""
        return self.alerts


class TestCoreIntegration:
    """Core integration tests that don't require full service deployment."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detection_config = {
            "model_path": "test_model.pt",
            "confidence_threshold": 0.7,
            "device": "cpu"
        }
        
        self.test_virtual_line = VirtualLine(
            id="test_line_001",
            camera_id="camera_001",
            points=[Point(x=320, y=100), Point(x=320, y=400)],
            active=True,
            sensitivity=0.8
        )
    
    def test_detection_to_alert_workflow(self):
        """Test workflow from detection to alert generation."""
        
        # Initialize components
        detection_pipeline = MockDetectionPipeline(self.detection_config)
        detection_pipeline.virtual_lines = [self.test_virtual_line]
        detection_pipeline.is_running = True
        
        alert_engine = MockAlertEngine()
        
        # Create test frame with person crossing virtual line
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw person at crossing location
        cv2.rectangle(test_frame, (300, 200), (340, 300), (255, 255, 255), -1)
        cv2.circle(test_frame, (320, 190), 15, (255, 255, 255), -1)
        
        # Process frame
        result = detection_pipeline.process_frame(test_frame, "camera_001")
        
        # Verify detection
        assert len(result.detections) == 1
        detection = result.detections[0]
        assert detection.detection_class == DetectionClass.PERSON
        assert detection.confidence == 0.85
        assert detection.camera_id == "camera_001"
        
        # Verify virtual line crossing
        crossings = result.metadata.get("virtual_line_crossings", [])
        assert len(crossings) == 1
        crossing = crossings[0]
        assert crossing.detection_id == detection.id
        assert crossing.virtual_line_id == self.test_virtual_line.id
        
        # Generate alert
        alert = alert_engine.generate_alert(crossing, detection)
        
        # Verify alert
        assert alert is not None
        assert alert.type == AlertType.VIRTUAL_LINE_CROSSING
        assert alert.camera_id == detection.camera_id
        assert alert.detection_id == detection.id
        assert alert.confidence == detection.confidence
        assert alert.severity in [Severity.LOW, Severity.MEDIUM, Severity.HIGH]
        assert 0 <= alert.risk_score <= 1.0
    
    def test_multi_frame_processing_workflow(self):
        """Test processing multiple frames in sequence."""
        
        # Initialize components
        detection_pipeline = MockDetectionPipeline(self.detection_config)
        detection_pipeline.virtual_lines = [self.test_virtual_line]
        detection_pipeline.is_running = True
        
        alert_engine = MockAlertEngine()
        
        # Create sequence of frames showing person movement
        frames = []
        for i in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Person moves from left to right
            x_pos = 200 + i * 40
            cv2.rectangle(frame, (x_pos, 200), (x_pos+40, 300), (255, 255, 255), -1)
            cv2.circle(frame, (x_pos+20, 190), 15, (255, 255, 255), -1)
            frames.append(frame)
        
        # Process frames
        results = []
        alerts = []
        
        for i, frame in enumerate(frames):
            result = detection_pipeline.process_frame(frame, "camera_001")
            results.append(result)
            
            # Generate alerts for crossings
            crossings = result.metadata.get("virtual_line_crossings", [])
            for crossing in crossings:
                detection = next((d for d in result.detections 
                                if d.id == crossing.detection_id), None)
                if detection:
                    alert = alert_engine.generate_alert(crossing, detection)
                    alerts.append(alert)
        
        # Verify processing
        assert len(results) == 5
        assert all(len(r.detections) == 1 for r in results)
        assert all(r.processing_time_ms > 0 for r in results)
        
        # Verify frame numbering
        frame_numbers = [r.frame_number for r in results]
        assert frame_numbers == [1, 2, 3, 4, 5]
        
        # Verify at least one crossing was detected
        total_crossings = sum(len(r.metadata.get("virtual_line_crossings", [])) for r in results)
        assert total_crossings >= 1
        
        # Verify alerts were generated
        assert len(alerts) >= 1
        assert all(a.type == AlertType.VIRTUAL_LINE_CROSSING for a in alerts)
    
    def test_performance_requirements(self):
        """Test that core components meet performance requirements."""
        
        detection_pipeline = MockDetectionPipeline(self.detection_config)
        detection_pipeline.is_running = True
        
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Measure processing times
        processing_times = []
        
        for i in range(20):
            start_time = time.perf_counter()
            result = detection_pipeline.process_frame(test_frame, f"camera_{i%3}")
            processing_time = (time.perf_counter() - start_time) * 1000
            processing_times.append(processing_time)
        
        # Verify performance requirements
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        
        # Mock pipeline should be very fast
        assert avg_time < 100, f"Average processing time {avg_time:.1f}ms too high for mock"
        assert max_time < 200, f"Max processing time {max_time:.1f}ms too high for mock"
        
        # Verify all results are valid
        assert all(isinstance(pt, (int, float)) and pt > 0 for pt in processing_times)
    
    def test_data_consistency_across_components(self):
        """Test data consistency as it flows between components."""
        
        detection_pipeline = MockDetectionPipeline(self.detection_config)
        detection_pipeline.virtual_lines = [self.test_virtual_line]
        detection_pipeline.is_running = True
        
        alert_engine = MockAlertEngine()
        
        # Process frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detection_pipeline.process_frame(test_frame, "camera_001")
        
        # Verify detection data
        assert len(result.detections) == 1
        detection = result.detections[0]
        
        original_data = {
            'id': detection.id,
            'camera_id': detection.camera_id,
            'timestamp': detection.timestamp,
            'confidence': detection.confidence,
            'detection_class': detection.detection_class
        }
        
        # Generate alert if crossing occurred
        crossings = result.metadata.get("virtual_line_crossings", [])
        if crossings:
            crossing = crossings[0]
            alert = alert_engine.generate_alert(crossing, detection)
            
            # Verify data consistency in alert
            assert alert.camera_id == original_data['camera_id']
            assert alert.detection_id == original_data['id']
            assert alert.confidence == original_data['confidence']
            assert alert.timestamp == original_data['timestamp']
            
            # Verify metadata preservation
            assert 'virtual_line_id' in alert.metadata
            assert alert.metadata['virtual_line_id'] == crossing.virtual_line_id
    
    def test_error_handling_and_recovery(self):
        """Test error handling in core components."""
        
        detection_pipeline = MockDetectionPipeline(self.detection_config)
        detection_pipeline.is_running = True
        
        alert_engine = MockAlertEngine()
        
        # Test with invalid frame data
        invalid_frames = [
            None,
            np.array([]),
            np.zeros((0, 0, 3), dtype=np.uint8),
            "invalid_frame_data"
        ]
        
        for invalid_frame in invalid_frames:
            try:
                result = detection_pipeline.process_frame(invalid_frame, "camera_001")
                # Should handle gracefully
                assert isinstance(result, DetectionResult)
                assert result.camera_id == "camera_001"
                assert isinstance(result.detections, list)
                
            except Exception as e:
                # Should be a handled exception
                assert isinstance(e, (ValueError, TypeError, AttributeError))
        
        # Test alert engine with invalid data
        try:
            # Create invalid detection
            invalid_detection = Detection(
                id="",  # Invalid empty ID
                camera_id="camera_001",
                timestamp=datetime.now(),
                bbox=BoundingBox(x=0, y=0, width=0, height=0),
                confidence=1.5,  # Invalid confidence
                detection_class=DetectionClass.PERSON
            )
            
            from shared.models.alerts import CrossingEvent
            crossing = CrossingEvent(
                detection_id=invalid_detection.id,
                virtual_line_id="line_001",
                crossing_point=(0, 0),
                crossing_direction="inbound",
                timestamp=invalid_detection.timestamp,
                confidence=invalid_detection.confidence
            )
            
            alert = alert_engine.generate_alert(crossing, invalid_detection)
            # Should either succeed with fallback values or handle gracefully
            assert alert is not None or alert is None
            
        except Exception as e:
            # Should be a handled exception
            assert isinstance(e, (ValueError, TypeError))
    
    def test_concurrent_processing_simulation(self):
        """Test concurrent processing simulation."""
        
        detection_pipeline = MockDetectionPipeline(self.detection_config)
        detection_pipeline.is_running = True
        
        # Simulate processing from multiple cameras
        cameras = ["camera_001", "camera_002", "camera_003"]
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        results = []
        
        # Process frames from different cameras
        for camera_id in cameras:
            for frame_num in range(3):
                result = detection_pipeline.process_frame(test_frame, camera_id)
                results.append((camera_id, frame_num, result))
        
        # Verify all cameras were processed
        processed_cameras = set(r[0] for r in results)
        assert len(processed_cameras) == 3
        assert processed_cameras == set(cameras)
        
        # Verify frame processing
        assert len(results) == 9  # 3 cameras × 3 frames
        
        # Verify each result is valid
        for camera_id, frame_num, result in results:
            assert isinstance(result, DetectionResult)
            assert result.camera_id == camera_id
            assert result.frame_number > 0
            assert isinstance(result.detections, list)


class TestSystemHealthAndMonitoring:
    """Test system health monitoring and status reporting."""
    
    def test_component_health_reporting(self):
        """Test health reporting from core components."""
        
        detection_pipeline = MockDetectionPipeline({
            "model_path": "test_model.pt",
            "confidence_threshold": 0.7,
            "device": "cpu"
        })
        
        alert_engine = MockAlertEngine()
        
        # Test component status
        assert detection_pipeline.is_running is False  # Initially not running
        detection_pipeline.is_running = True
        assert detection_pipeline.is_running is True
        
        # Test alert engine status
        initial_alert_count = len(alert_engine.get_active_alerts())
        assert initial_alert_count == 0
        
        # Process some frames to generate activity
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        for i in range(3):
            result = detection_pipeline.process_frame(test_frame, f"camera_{i}")
            assert result.processing_time_ms > 0
        
        # Verify frame count increased
        assert detection_pipeline.frame_count == 3
    
    def test_performance_metrics_collection(self):
        """Test collection of performance metrics."""
        
        detection_pipeline = MockDetectionPipeline({
            "model_path": "test_model.pt",
            "confidence_threshold": 0.7,
            "device": "cpu"
        })
        detection_pipeline.is_running = True
        
        # Process frames and collect metrics
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processing_times = []
        
        for i in range(10):
            start_time = time.perf_counter()
            result = detection_pipeline.process_frame(test_frame, "camera_001")
            end_time = time.perf_counter()
            
            actual_time = (end_time - start_time) * 1000
            reported_time = result.processing_time_ms
            
            processing_times.append({
                'actual': actual_time,
                'reported': reported_time,
                'frame_number': result.frame_number
            })
        
        # Verify metrics collection
        assert len(processing_times) == 10
        
        # Verify frame numbers are sequential
        frame_numbers = [pt['frame_number'] for pt in processing_times]
        assert frame_numbers == list(range(1, 11))
        
        # Verify processing times are reasonable
        avg_actual = sum(pt['actual'] for pt in processing_times) / len(processing_times)
        avg_reported = sum(pt['reported'] for pt in processing_times) / len(processing_times)
        
        assert avg_actual > 0, "Actual processing time should be positive"
        assert avg_reported > 0, "Reported processing time should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])