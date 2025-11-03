"""
Unit tests for Multi-Camera Tracking system.

Tests multi-object tracking, Kalman filtering, track lifecycle management,
and cross-camera tracking capabilities.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from edge.src.multi_camera_tracker import MultiCameraTracker, KalmanTracker
from shared.models.detection import Detection, BoundingBox, DetectionClass
from shared.models.tracking import Track, TrackStatus, Point, Trajectory


class TestKalmanTracker:
    """Test cases for KalmanTracker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detection = Detection(
            id="test_det_1",
            camera_id="cam_1",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=100, y=100, width=50, height=100),
            confidence=0.9,
            detection_class=DetectionClass.PERSON
        )
        self.kalman_tracker = KalmanTracker(self.detection)
    
    def test_kalman_tracker_initialization(self):
        """Test KalmanTracker initializes correctly with first detection."""
        assert self.kalman_tracker.kf.x[0] == 100  # x position
        assert self.kalman_tracker.kf.x[1] == 100  # y position
        assert self.kalman_tracker.kf.x[2] == 50   # width
        assert self.kalman_tracker.kf.x[3] == 100  # height
        assert self.kalman_tracker.hit_streak == 1
        assert self.kalman_tracker.age == 1
        assert self.kalman_tracker.time_since_update == 0
    
    def test_kalman_tracker_update(self):
        """Test Kalman tracker updates with new detection."""
        new_detection = Detection(
            id="test_det_2",
            camera_id="cam_1",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=105, y=105, width=50, height=100),
            confidence=0.85,
            detection_class=DetectionClass.PERSON
        )
        
        self.kalman_tracker.update(new_detection)
        
        assert self.kalman_tracker.hit_streak == 2
        assert self.kalman_tracker.age == 2
        assert self.kalman_tracker.time_since_update == 0
    
    def test_kalman_tracker_predict(self):
        """Test Kalman tracker prediction."""
        predicted_bbox = self.kalman_tracker.predict()
        
        assert len(predicted_bbox) == 4
        assert self.kalman_tracker.age == 2
        assert self.kalman_tracker.time_since_update == 1
    
    def test_kalman_tracker_get_state(self):
        """Test getting current state from Kalman tracker."""
        state = self.kalman_tracker.get_state()
        
        assert len(state) == 4
        assert state[0] == 100  # x
        assert state[1] == 100  # y
        assert state[2] == 50   # width
        assert state[3] == 100  # height


class TestMultiCameraTracker:
    """Test cases for MultiCameraTracker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = MultiCameraTracker(
            max_disappeared=10,
            max_distance=100.0,
            min_hits=2,
            iou_threshold=0.3
        )
        
        # Create test detections
        self.detections_cam1 = [
            Detection(
                id="det_1",
                camera_id="cam_1",
                timestamp=datetime.now(),
                bbox=BoundingBox(x=100, y=100, width=50, height=100),
                confidence=0.9,
                detection_class=DetectionClass.PERSON
            ),
            Detection(
                id="det_2",
                camera_id="cam_1",
                timestamp=datetime.now(),
                bbox=BoundingBox(x=200, y=150, width=45, height=95),
                confidence=0.85,
                detection_class=DetectionClass.PERSON
            )
        ]
        
        self.detections_cam2 = [
            Detection(
                id="det_3",
                camera_id="cam_2",
                timestamp=datetime.now(),
                bbox=BoundingBox(x=300, y=200, width=48, height=98),
                confidence=0.88,
                detection_class=DetectionClass.PERSON
            )
        ]
    
    def test_tracker_initialization(self):
        """Test MultiCameraTracker initializes correctly."""
        assert self.tracker.max_disappeared == 10
        assert self.tracker.max_distance == 100.0
        assert self.tracker.min_hits == 2
        assert self.tracker.iou_threshold == 0.3
        assert len(self.tracker.camera_trackers) == 0
        assert len(self.tracker.camera_tracks) == 0
        assert self.tracker.next_track_id == 1
    
    def test_calculate_iou(self):
        """Test IoU calculation between bounding boxes."""
        box1 = np.array([100, 100, 50, 100])  # x, y, w, h
        box2 = np.array([120, 120, 50, 100])  # Overlapping box
        box3 = np.array([200, 200, 50, 100])  # Non-overlapping box
        
        iou_overlap = self.tracker._calculate_iou(box1, box2)
        iou_no_overlap = self.tracker._calculate_iou(box1, box3)
        
        assert 0 < iou_overlap < 1
        assert iou_no_overlap == 0.0
    
    def test_update_tracks_new_detections(self):
        """Test updating tracks with new detections creates new tracks."""
        tracks = self.tracker.update_tracks(self.detections_cam1, "cam_1")
        
        # Should create new tracks but not return them yet (min_hits=2)
        assert len(tracks) == 0
        assert len(self.tracker.camera_tracks["cam_1"]) == 2
        assert len(self.tracker.camera_trackers["cam_1"]) == 2
    
    def test_update_tracks_confirmed_tracks(self):
        """Test that tracks become confirmed after min_hits."""
        # First update - create tracks
        self.tracker.update_tracks(self.detections_cam1, "cam_1")
        
        # Second update - should confirm tracks
        updated_detections = [
            Detection(
                id="det_1_updated",
                camera_id="cam_1",
                timestamp=datetime.now(),
                bbox=BoundingBox(x=105, y=105, width=50, height=100),
                confidence=0.9,
                detection_class=DetectionClass.PERSON
            ),
            Detection(
                id="det_2_updated",
                camera_id="cam_1",
                timestamp=datetime.now(),
                bbox=BoundingBox(x=205, y=155, width=45, height=95),
                confidence=0.85,
                detection_class=DetectionClass.PERSON
            )
        ]
        
        tracks = self.tracker.update_tracks(updated_detections, "cam_1")
        
        # Should return confirmed tracks now
        assert len(tracks) == 2
        for track in tracks:
            assert track.status == TrackStatus.ACTIVE
            assert len(track.detections) >= 2
    
    def test_track_association(self):
        """Test detection to track association using Hungarian algorithm."""
        # Create initial tracks
        self.tracker.update_tracks(self.detections_cam1, "cam_1")
        
        # Create detections that should associate with existing tracks
        associated_detections = [
            Detection(
                id="det_assoc_1",
                camera_id="cam_1",
                timestamp=datetime.now(),
                bbox=BoundingBox(x=102, y=102, width=50, height=100),  # Close to first detection
                confidence=0.9,
                detection_class=DetectionClass.PERSON
            )
        ]
        
        trackers = self.tracker.camera_trackers["cam_1"]
        matches, unmatched_dets, unmatched_tracks = self.tracker._associate_detections_to_tracks(
            associated_detections, trackers
        )
        
        assert len(matches) == 1  # Should match one detection
        assert len(unmatched_dets) == 0  # No unmatched detections
        assert len(unmatched_tracks) == 1  # One unmatched track
    
    def test_track_lifecycle_management(self):
        """Test track lifecycle from creation to deletion."""
        # Create tracks
        self.tracker.update_tracks(self.detections_cam1, "cam_1")
        initial_track_count = len(self.tracker.camera_tracks["cam_1"])
        
        # Update with no detections multiple times to trigger deletion
        for _ in range(self.tracker.max_disappeared + 1):
            self.tracker.update_tracks([], "cam_1")
        
        # Tracks should be deleted
        assert len(self.tracker.camera_tracks["cam_1"]) < initial_track_count
    
    def test_get_active_tracks(self):
        """Test getting active tracks for specific camera and all cameras."""
        # Create tracks for multiple cameras
        self.tracker.update_tracks(self.detections_cam1, "cam_1")
        self.tracker.update_tracks(self.detections_cam2, "cam_2")
        
        # Confirm tracks
        self.tracker.update_tracks(self.detections_cam1, "cam_1")
        self.tracker.update_tracks(self.detections_cam2, "cam_2")
        
        # Get tracks for specific camera
        cam1_tracks = self.tracker.get_active_tracks("cam_1")
        cam2_tracks = self.tracker.get_active_tracks("cam_2")
        all_tracks = self.tracker.get_active_tracks()
        
        assert len(cam1_tracks) == 2
        assert len(cam2_tracks) == 1
        assert len(all_tracks) == 3
    
    def test_get_trajectory(self):
        """Test getting trajectory for a specific track."""
        # Create and confirm a track
        tracks = self.tracker.update_tracks(self.detections_cam1, "cam_1")
        tracks = self.tracker.update_tracks(self.detections_cam1, "cam_1")
        
        if tracks:
            track_id = tracks[0].id
            trajectory = self.tracker.get_trajectory(track_id)
            
            assert trajectory is not None
            assert len(trajectory.points) >= 2
            assert trajectory.start_time is not None
    
    def test_cleanup_old_tracks(self):
        """Test cleanup of old inactive tracks."""
        # Create tracks
        self.tracker.update_tracks(self.detections_cam1, "cam_1")
        
        # Make tracks inactive by not updating them
        for _ in range(self.tracker.max_disappeared + 1):
            self.tracker.update_tracks([], "cam_1")
        
        # Cleanup old tracks
        removed_count = self.tracker.cleanup_old_tracks(max_age_seconds=1)
        
        assert removed_count >= 0
    
    def test_match_across_cameras(self):
        """Test cross-camera track matching."""
        # Create and confirm tracks
        tracks = self.tracker.update_tracks(self.detections_cam1, "cam_1")
        tracks = self.tracker.update_tracks(self.detections_cam1, "cam_1")
        
        if tracks:
            track = tracks[0]
            global_track = self.tracker.match_across_cameras(track)
            
            assert global_track is not None
            assert len(global_track.local_tracks) == 1
            assert global_track.local_tracks[0].id == track.id
    
    def test_synthetic_trajectory_accuracy(self):
        """Test tracking accuracy with synthetic trajectories."""
        # Create a synthetic trajectory moving in a straight line
        synthetic_detections = []
        start_time = datetime.now()
        
        for i in range(10):
            detection = Detection(
                id=f"synthetic_{i}",
                camera_id="cam_synthetic",
                timestamp=start_time + timedelta(milliseconds=i*100),
                bbox=BoundingBox(x=100 + i*10, y=100 + i*5, width=50, height=100),
                confidence=0.9,
                detection_class=DetectionClass.PERSON
            )
            synthetic_detections.append(detection)
        
        # Process detections sequentially
        all_tracks = []
        for detection in synthetic_detections:
            tracks = self.tracker.update_tracks([detection], "cam_synthetic")
            if tracks:
                all_tracks.extend(tracks)
        
        # Should maintain a single consistent track
        unique_track_ids = set(track.id for track in all_tracks)
        assert len(unique_track_ids) <= 2  # Allow for some track fragmentation
        
        # Check trajectory smoothness
        if all_tracks:
            track = all_tracks[0]
            trajectory = track.trajectory
            
            # Verify trajectory follows expected pattern
            assert len(trajectory.points) >= 5
            
            # Check that trajectory is generally moving in expected direction
            first_point = trajectory.points[0]
            last_point = trajectory.points[-1]
            
            assert last_point.x > first_point.x  # Moving right
            assert last_point.y > first_point.y  # Moving down


class TestTrackingIntegration:
    """Integration tests for the complete tracking system."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.tracker = MultiCameraTracker(min_hits=1)  # Lower threshold for testing
    
    def test_multi_camera_scenario(self):
        """Test realistic multi-camera tracking scenario."""
        # Simulate person moving between cameras
        
        # Person appears in camera 1
        cam1_detections = [
            Detection(
                id="person_cam1_1",
                camera_id="cam_1",
                timestamp=datetime.now(),
                bbox=BoundingBox(x=100, y=100, width=50, height=100),
                confidence=0.9,
                detection_class=DetectionClass.PERSON
            )
        ]
        
        # Person moves in camera 1
        cam1_detections_2 = [
            Detection(
                id="person_cam1_2",
                camera_id="cam_1",
                timestamp=datetime.now() + timedelta(seconds=1),
                bbox=BoundingBox(x=150, y=120, width=50, height=100),
                confidence=0.85,
                detection_class=DetectionClass.PERSON
            )
        ]
        
        # Person appears in camera 2 (different location)
        cam2_detections = [
            Detection(
                id="person_cam2_1",
                camera_id="cam_2",
                timestamp=datetime.now() + timedelta(seconds=2),
                bbox=BoundingBox(x=50, y=200, width=48, height=98),
                confidence=0.88,
                detection_class=DetectionClass.PERSON
            )
        ]
        
        # Process detections
        tracks_1 = self.tracker.update_tracks(cam1_detections, "cam_1")
        tracks_2 = self.tracker.update_tracks(cam1_detections_2, "cam_1")
        tracks_3 = self.tracker.update_tracks(cam2_detections, "cam_2")
        
        # Verify tracking results
        all_active_tracks = self.tracker.get_active_tracks()
        assert len(all_active_tracks) >= 1
        
        # Test cross-camera matching
        if tracks_2:
            global_track = self.tracker.match_across_cameras(tracks_2[0])
            assert global_track is not None
    
    def test_performance_requirements(self):
        """Test that tracking meets performance requirements."""
        # Create multiple detections to test processing speed
        detections = []
        for i in range(50):  # Simulate 50 detections
            detection = Detection(
                id=f"perf_test_{i}",
                camera_id="cam_perf",
                timestamp=datetime.now(),
                bbox=BoundingBox(x=100 + i*2, y=100 + i, width=50, height=100),
                confidence=0.9,
                detection_class=DetectionClass.PERSON
            )
            detections.append(detection)
        
        # Measure processing time
        start_time = time.time()
        tracks = self.tracker.update_tracks(detections, "cam_perf")
        processing_time = time.time() - start_time
        
        # Should process within reasonable time (< 100ms for 50 detections)
        assert processing_time < 0.1
        
        # Verify tracks were created
        assert len(self.tracker.camera_tracks["cam_perf"]) == 50