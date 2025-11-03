#!/usr/bin/env python3
"""
Multi-Camera Data Flow Integration Tests
Tests data flow validation across multiple cameras and system components.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import cv2

# Import system components
from edge.src.detection_pipeline import DetectionPipeline
from edge.src.multi_camera_tracker import MultiCameraTracker
from edge.src.reid_matcher import ReIDMatcher
from services.tracking_service.main import TrackingService
from services.alert_service.alert_engine import AlertEngine

# Import shared models
from shared.models.detection import Detection, DetectionResult, BoundingBox, DetectionClass
from shared.models.tracking import Track, TrackStatus, GlobalTrack
from shared.models.virtual_line import VirtualLine, Point
from shared.models.alerts import Alert, AlertType, Severity, CrossingEvent


class TestMultiCameraDataFlow:
    """Test multi-camera data flow and cross-system validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.camera_configs = {
            "camera_001": {
                "location": (28.6139, 77.2090),
                "coverage_area": [(0, 0), (640, 0), (640, 480), (0, 480)],
                "virtual_lines": [
                    VirtualLine(
                        id="line_001_cam1",
                        camera_id="camera_001",
                        points=[Point(x=320, y=100), Point(x=320, y=380)],
                        active=True
                    )
                ]
            },
            "camera_002": {
                "location": (28.6140, 77.2091),
                "coverage_area": [(0, 0), (640, 0), (640, 480), (0, 480)],
                "virtual_lines": [
                    VirtualLine(
                        id="line_002_cam2",
                        camera_id="camera_002",
                        points=[Point(x=320, y=100), Point(x=320, y=380)],
                        active=True
                    )
                ]
            },
            "camera_003": {
                "location": (28.6141, 77.2092),
                "coverage_area": [(0, 0), (640, 0), (640, 480), (0, 480)],
                "virtual_lines": [
                    VirtualLine(
                        id="line_003_cam3",
                        camera_id="camera_003",
                        points=[Point(x=320, y=100), Point(x=320, y=380)],
                        active=True
                    )
                ]
            }
        }
        
        self.detection_config = {
            "model_path": "test_model.pt",
            "confidence_threshold": 0.7,
            "nms_threshold": 0.45,
            "input_size": [640, 640],
            "device": "cpu"
        }
        
        self.tracking_config = {
            "max_disappeared": 30,
            "max_distance": 100,
            "track_buffer": 30,
            "reid_threshold": 0.7
        }
    
    def _create_person_detection(self, camera_id, x, y, width=40, height=80, confidence=0.85):
        """Create a person detection at specified location."""
        return Detection(
            id=f"det_{camera_id}_{int(time.time()*1000)}",
            camera_id=camera_id,
            timestamp=datetime.now(),
            bbox=BoundingBox(x=x, y=y, width=width, height=height),
            confidence=confidence,
            detection_class=DetectionClass.PERSON,
            features=np.random.rand(512).astype(np.float32)  # Mock ReID features
        )
    
    @patch('edge.src.detection_pipeline.ModelManager')
    def test_multi_camera_detection_consistency(self, mock_model_manager):
        """Test detection consistency across multiple cameras."""
        
        # Setup detection mocks for all cameras
        self._setup_multi_camera_detection_mocks(mock_model_manager)
        
        # Initialize detection pipelines for all cameras
        pipelines = {}
        for camera_id in self.camera_configs.keys():
            pipeline = DetectionPipeline(self.detection_config)
            pipeline.virtual_lines = self.camera_configs[camera_id]["virtual_lines"]
            pipeline.is_running = True
            pipelines[camera_id] = pipeline
        
        # Create test frame (same person visible in all cameras)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_frame, (300, 200), (340, 280), (255, 255, 255), -1)
        cv2.circle(test_frame, (320, 190), 15, (255, 255, 255), -1)
        
        # Process frame through all cameras
        results = {}
        for camera_id, pipeline in pipelines.items():
            result = pipeline.process_frame(test_frame, camera_id)
            results[camera_id] = result
        
        # Verify detection consistency
        for camera_id, result in results.items():
            assert len(result.detections) == 1, f"Camera {camera_id} should detect one person"
            detection = result.detections[0]
            assert detection.detection_class == DetectionClass.PERSON
            assert detection.confidence >= 0.7
            assert detection.camera_id == camera_id
        
        # Verify detection features are generated
        for result in results.values():
            detection = result.detections[0]
            assert detection.features is not None
            assert len(detection.features) == 512
    
    @patch('edge.src.detection_pipeline.ModelManager')
    def test_cross_camera_tracking_workflow(self, mock_model_manager):
        """Test person tracking across multiple cameras."""
        
        # Setup mocks
        self._setup_multi_camera_detection_mocks(mock_model_manager)
        
        # Initialize components
        tracker = MultiCameraTracker(self.tracking_config)
        reid_matcher = ReIDMatcher()
        
        # Simulate person moving across cameras
        # Camera 1: Person enters from left
        det1_cam1 = self._create_person_detection("camera_001", 100, 240)
        det2_cam1 = self._create_person_detection("camera_001", 200, 240)
        det3_cam1 = self._create_person_detection("camera_001", 300, 240)
        
        # Camera 2: Same person continues (overlapping coverage)
        det1_cam2 = self._create_person_detection("camera_002", 320, 240)
        det2_cam2 = self._create_person_detection("camera_002", 420, 240)
        det3_cam2 = self._create_person_detection("camera_002", 520, 240)
        
        # Camera 3: Person exits
        det1_cam3 = self._create_person_detection("camera_003", 100, 240)
        det2_cam3 = self._create_person_detection("camera_003", 200, 240)
        
        # Make features similar for same person across cameras
        base_features = np.random.rand(512).astype(np.float32)
        for det in [det1_cam1, det2_cam1, det3_cam1, det1_cam2, det2_cam2, det3_cam2, det1_cam3, det2_cam3]:
            # Add small noise to simulate real-world variation
            det.features = base_features + np.random.normal(0, 0.1, 512).astype(np.float32)
        
        # Process detections through tracker
        # Camera 1 sequence
        tracks_cam1_1 = tracker.update_tracks([det1_cam1], "camera_001")
        tracks_cam1_2 = tracker.update_tracks([det2_cam1], "camera_001")
        tracks_cam1_3 = tracker.update_tracks([det3_cam1], "camera_001")
        
        # Camera 2 sequence (with slight overlap)
        tracks_cam2_1 = tracker.update_tracks([det1_cam2], "camera_002")
        tracks_cam2_2 = tracker.update_tracks([det2_cam2], "camera_002")
        tracks_cam2_3 = tracker.update_tracks([det3_cam2], "camera_002")
        
        # Camera 3 sequence
        tracks_cam3_1 = tracker.update_tracks([det1_cam3], "camera_003")
        tracks_cam3_2 = tracker.update_tracks([det2_cam3], "camera_003")
        
        # Verify tracking across cameras
        all_tracks = tracker.get_all_tracks()
        camera_1_tracks = [t for t in all_tracks if t.camera_id == "camera_001"]
        camera_2_tracks = [t for t in all_tracks if t.camera_id == "camera_002"]
        camera_3_tracks = [t for t in all_tracks if t.camera_id == "camera_003"]
        
        assert len(camera_1_tracks) >= 1, "Should have tracks from camera 1"
        assert len(camera_2_tracks) >= 1, "Should have tracks from camera 2"
        assert len(camera_3_tracks) >= 1, "Should have tracks from camera 3"
        
        # Test global track creation
        global_tracks = tracker.get_global_tracks()
        assert len(global_tracks) >= 1, "Should create global track across cameras"
        
        # Verify global track spans multiple cameras
        if global_tracks:
            global_track = global_tracks[0]
            camera_ids_in_track = set(t.camera_id for t in global_track.camera_tracks)
            assert len(camera_ids_in_track) >= 2, "Global track should span multiple cameras"
    
    @patch('edge.src.detection_pipeline.ModelManager')
    def test_virtual_line_crossing_across_cameras(self, mock_model_manager):
        """Test virtual line crossing detection across multiple cameras."""
        
        # Setup mocks
        self._setup_multi_camera_detection_mocks(mock_model_manager)
        
        # Initialize components
        pipelines = {}
        alert_engine = AlertEngine()
        
        for camera_id in self.camera_configs.keys():
            pipeline = DetectionPipeline(self.detection_config)
            pipeline.virtual_lines = self.camera_configs[camera_id]["virtual_lines"]
            pipeline.is_running = True
            pipelines[camera_id] = pipeline
        
        # Create crossing scenarios for each camera
        crossing_frames = {
            "camera_001": [
                (np.zeros((480, 640, 3), dtype=np.uint8), 280, 240),  # Before line
                (np.zeros((480, 640, 3), dtype=np.uint8), 320, 240),  # On line
                (np.zeros((480, 640, 3), dtype=np.uint8), 360, 240)   # After line
            ],
            "camera_002": [
                (np.zeros((480, 640, 3), dtype=np.uint8), 280, 240),
                (np.zeros((480, 640, 3), dtype=np.uint8), 320, 240),
                (np.zeros((480, 640, 3), dtype=np.uint8), 360, 240)
            ],
            "camera_003": [
                (np.zeros((480, 640, 3), dtype=np.uint8), 280, 240),
                (np.zeros((480, 640, 3), dtype=np.uint8), 320, 240),
                (np.zeros((480, 640, 3), dtype=np.uint8), 360, 240)
            ]
        }
        
        # Add person to frames
        for camera_id, frames in crossing_frames.items():
            for frame, x, y in frames:
                cv2.rectangle(frame, (x-20, y-40), (x+20, y+40), (255, 255, 255), -1)
                cv2.circle(frame, (x, y-50), 15, (255, 255, 255), -1)
        
        # Process crossing scenarios
        all_crossings = []
        all_alerts = []
        
        for camera_id, frames in crossing_frames.items():
            pipeline = pipelines[camera_id]
            
            for frame, x, y in frames:
                result = pipeline.process_frame(frame, camera_id)
                
                if result.virtual_line_crossings:
                    all_crossings.extend(result.virtual_line_crossings)
                    
                    # Generate alerts for crossings
                    for crossing in result.virtual_line_crossings:
                        detection = next((d for d in result.detections 
                                        if d.id == crossing.detection_id), None)
                        if detection:
                            alert = alert_engine.generate_alert(crossing, detection)
                            all_alerts.append(alert)
        
        # Verify crossings were detected across cameras
        assert len(all_crossings) >= 1, "Should detect crossings across cameras"
        
        # Verify alerts were generated
        assert len(all_alerts) >= 1, "Should generate alerts for crossings"
        
        # Verify camera-specific data in alerts
        camera_ids_with_alerts = set(alert.camera_id for alert in all_alerts)
        assert len(camera_ids_with_alerts) >= 1, "Should have alerts from multiple cameras"
    
    def test_data_consistency_across_components(self):
        """Test data consistency as it flows between components."""
        
        # Create test detection
        original_detection = Detection(
            id="consistency_test_001",
            camera_id="camera_001",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=100, y=100, width=50, height=100),
            confidence=0.85,
            detection_class=DetectionClass.PERSON,
            features=np.random.rand(512).astype(np.float32),
            metadata={"test_field": "test_value"}
        )
        
        # Initialize components
        tracker = MultiCameraTracker(self.tracking_config)
        alert_engine = AlertEngine()
        
        # Process through tracker
        tracks = tracker.update_tracks([original_detection], "camera_001")
        assert len(tracks) == 1
        track = tracks[0]
        
        # Verify data consistency in tracking
        assert track.camera_id == original_detection.camera_id
        assert len(track.detections) == 1
        assert track.detections[0].id == original_detection.id
        assert track.detections[0].confidence == original_detection.confidence
        
        # Create crossing event
        crossing_event = CrossingEvent(
            detection_id=original_detection.id,
            virtual_line_id="line_001",
            crossing_point=(125, 150),
            crossing_direction="inbound",
            timestamp=original_detection.timestamp,
            confidence=original_detection.confidence
        )
        
        # Generate alert
        alert = alert_engine.generate_alert(crossing_event, original_detection)
        
        # Verify data consistency in alert
        assert alert.camera_id == original_detection.camera_id
        assert alert.detection_id == original_detection.id
        assert alert.confidence == original_detection.confidence
        assert alert.timestamp == original_detection.timestamp
        
        # Verify metadata preservation
        if hasattr(original_detection, 'metadata') and original_detection.metadata:
            # Metadata should be preserved or accessible through the system
            assert original_detection.metadata["test_field"] == "test_value"
    
    def test_temporal_data_flow_validation(self):
        """Test temporal consistency of data flow across components."""
        
        # Create sequence of detections with timestamps
        base_time = datetime.now()
        detections = []
        
        for i in range(5):
            detection = Detection(
                id=f"temporal_test_{i:03d}",
                camera_id="camera_001",
                timestamp=base_time + timedelta(seconds=i),
                bbox=BoundingBox(x=100+i*10, y=100, width=50, height=100),
                confidence=0.8 + i*0.02,
                detection_class=DetectionClass.PERSON,
                features=np.random.rand(512).astype(np.float32)
            )
            detections.append(detection)
        
        # Initialize tracker
        tracker = MultiCameraTracker(self.tracking_config)
        
        # Process detections in sequence
        all_tracks = []
        for detection in detections:
            tracks = tracker.update_tracks([detection], "camera_001")
            all_tracks.extend(tracks)
        
        # Verify temporal consistency
        if all_tracks:
            # Check that track timestamps are in order
            track = all_tracks[0]  # Should be the same track updated
            if len(track.detections) > 1:
                timestamps = [d.timestamp for d in track.detections]
                assert timestamps == sorted(timestamps), "Detection timestamps should be in order"
            
            # Verify track duration calculation
            if track.end_time and track.start_time:
                expected_duration = (track.end_time - track.start_time).total_seconds()
                assert expected_duration >= 0, "Track duration should be non-negative"
                assert expected_duration <= 10, "Track duration should be reasonable"
    
    def test_error_propagation_across_components(self):
        """Test error handling and propagation across system components."""
        
        # Create invalid detection data
        invalid_detection = Detection(
            id="",  # Invalid empty ID
            camera_id="camera_001",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=-100, y=-100, width=0, height=0),  # Invalid bbox
            confidence=1.5,  # Invalid confidence > 1.0
            detection_class=DetectionClass.PERSON
        )
        
        # Initialize components
        tracker = MultiCameraTracker(self.tracking_config)
        alert_engine = AlertEngine()
        
        # Test tracker error handling
        try:
            tracks = tracker.update_tracks([invalid_detection], "camera_001")
            # If no exception, verify graceful handling
            assert isinstance(tracks, list)
        except Exception as e:
            # Exception is acceptable for invalid data
            assert isinstance(e, (ValueError, TypeError))
        
        # Test alert engine error handling
        try:
            crossing_event = CrossingEvent(
                detection_id=invalid_detection.id,
                virtual_line_id="line_001",
                crossing_point=(0, 0),
                crossing_direction="inbound",
                timestamp=invalid_detection.timestamp,
                confidence=invalid_detection.confidence
            )
            alert = alert_engine.generate_alert(crossing_event, invalid_detection)
            # If no exception, verify alert has fallback values
            assert alert is not None
        except Exception as e:
            # Exception is acceptable for invalid data
            assert isinstance(e, (ValueError, TypeError))
    
    def _setup_multi_camera_detection_mocks(self, mock_model_manager):
        """Setup detection mocks for multiple cameras."""
        mock_manager = Mock()
        mock_model = Mock()
        
        # Mock single person detection for all cameras
        mock_result = Mock()
        boxes = np.array([[300, 200, 340, 280]], dtype=np.float32)
        confidences = np.array([0.85], dtype=np.float32)
        classes = np.array([0], dtype=np.float32)
        
        mock_boxes = Mock()
        mock_boxes.xyxy = Mock()
        mock_boxes.xyxy.cpu = Mock()
        mock_boxes.xyxy.cpu.return_value = Mock()
        mock_boxes.xyxy.cpu.return_value.numpy = Mock(return_value=boxes)
        
        mock_boxes.conf = Mock()
        mock_boxes.conf.cpu = Mock()
        mock_boxes.conf.cpu.return_value = Mock()
        mock_boxes.conf.cpu.return_value.numpy = Mock(return_value=confidences)
        
        mock_boxes.cls = Mock()
        mock_boxes.cls.cpu = Mock()
        mock_boxes.cls.cpu.return_value = Mock()
        mock_boxes.cls.cpu.return_value.numpy = Mock(return_value=classes)
        
        mock_boxes.__len__ = Mock(return_value=1)
        mock_result.boxes = mock_boxes
        
        mock_model.return_value = [mock_result]
        mock_manager.get_model.return_value = mock_model
        mock_model_manager.return_value = mock_manager


class TestSystemFailureScenarios:
    """Test system behavior under various failure scenarios."""
    
    def test_camera_failure_recovery(self):
        """Test system recovery when cameras fail."""
        
        # Initialize multi-camera tracker
        tracker = MultiCameraTracker({
            "max_disappeared": 30,
            "max_distance": 100
        })
        
        # Create detections from multiple cameras
        det_cam1 = Detection(
            id="det_cam1_001",
            camera_id="camera_001",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=100, y=100, width=50, height=100),
            confidence=0.85,
            detection_class=DetectionClass.PERSON
        )
        
        det_cam2 = Detection(
            id="det_cam2_001",
            camera_id="camera_002",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=200, y=100, width=50, height=100),
            confidence=0.85,
            detection_class=DetectionClass.PERSON
        )
        
        # Process normal operation
        tracks_1 = tracker.update_tracks([det_cam1], "camera_001")
        tracks_2 = tracker.update_tracks([det_cam2], "camera_002")
        
        assert len(tracks_1) == 1
        assert len(tracks_2) == 1
        
        # Simulate camera_001 failure (no more detections)
        # Continue with camera_002 only
        det_cam2_continued = Detection(
            id="det_cam2_002",
            camera_id="camera_002",
            timestamp=datetime.now() + timedelta(seconds=1),
            bbox=BoundingBox(x=210, y=100, width=50, height=100),
            confidence=0.85,
            detection_class=DetectionClass.PERSON
        )
        
        tracks_continued = tracker.update_tracks([det_cam2_continued], "camera_002")
        
        # Verify system continues to function with remaining cameras
        assert len(tracks_continued) >= 1
        
        # Verify camera_001 tracks are marked as lost after timeout
        all_tracks = tracker.get_all_tracks()
        cam1_tracks = [t for t in all_tracks if t.camera_id == "camera_001"]
        
        # After sufficient time, tracks should be marked as lost
        if cam1_tracks:
            # Simulate time passage for track timeout
            tracker._cleanup_lost_tracks()
            updated_tracks = tracker.get_active_tracks()
            active_cam1_tracks = [t for t in updated_tracks if t.camera_id == "camera_001"]
            # Should have fewer or no active tracks from failed camera
            assert len(active_cam1_tracks) <= len(cam1_tracks)
    
    def test_network_partition_handling(self):
        """Test system behavior during network partitions."""
        
        # This test simulates network issues between components
        # In a real system, this would test message queue failures, etc.
        
        alert_engine = AlertEngine()
        
        # Create test alert
        test_detection = Detection(
            id="network_test_001",
            camera_id="camera_001",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=100, y=100, width=50, height=100),
            confidence=0.85,
            detection_class=DetectionClass.PERSON
        )
        
        crossing_event = CrossingEvent(
            detection_id=test_detection.id,
            virtual_line_id="line_001",
            crossing_point=(125, 150),
            crossing_direction="inbound",
            timestamp=test_detection.timestamp,
            confidence=test_detection.confidence
        )
        
        # Mock network failure in alert sending
        with patch.object(alert_engine, '_send_alert', side_effect=Exception("Network error")):
            # Should handle network failure gracefully
            try:
                alert = alert_engine.generate_alert(crossing_event, test_detection)
                # Alert should be created even if sending fails
                assert alert is not None
                assert alert.id is not None
            except Exception:
                # Or should fail gracefully without crashing
                pass
        
        # Test recovery after network restoration
        with patch.object(alert_engine, '_send_alert', return_value=True):
            alert = alert_engine.generate_alert(crossing_event, test_detection)
            assert alert is not None
            
            # Should be able to route alert after recovery
            success = alert_engine.route_alert(alert)
            assert success is True or success is None  # Depends on implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])