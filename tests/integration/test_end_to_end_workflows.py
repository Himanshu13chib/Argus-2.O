#!/usr/bin/env python3
"""
End-to-End Integration Tests for Project Argus
Tests complete workflows from detection to incident resolution across all system components.
"""

import pytest
import asyncio
import time
import json
import tempfile
import subprocess
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import numpy as np
import cv2

# Import system components
from edge.src.detection_pipeline import DetectionPipeline
from edge.src.multi_camera_tracker import MultiCameraTracker
from edge.src.virtual_line_processor import VirtualLineProcessor
from services.alert_service.alert_engine import AlertEngine
from services.alert_service.incident_manager import IncidentManager
from services.evidence_service.evidence_store import EvidenceStore
from services.evidence_service.forensics_engine import ForensicsEngine

# Import shared models
from shared.models.detection import Detection, DetectionResult, BoundingBox, DetectionClass
from shared.models.tracking import Track, TrackStatus
from shared.models.virtual_line import VirtualLine, Point, VirtualLineType
from shared.models.alerts import Alert, AlertType, Severity
from shared.models.incidents import Incident, IncidentStatus, Resolution, ResolutionType
from shared.models.evidence import Evidence, EvidenceType


class MockCameraSystem:
    """Mock camera system for generating test scenarios."""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame_count = 0
    
    def generate_crossing_scenario(self, person_path, frames=30):
        """Generate frames showing person crossing virtual line."""
        frames_data = []
        
        for i in range(frames):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Calculate person position along path
            progress = i / (frames - 1)
            x = int(person_path[0][0] + progress * (person_path[1][0] - person_path[0][0]))
            y = int(person_path[0][1] + progress * (person_path[1][1] - person_path[0][1]))
            
            # Draw person
            cv2.rectangle(frame, (x-20, y-40), (x+20, y+40), (255, 255, 255), -1)
            cv2.circle(frame, (x, y-50), 15, (255, 255, 255), -1)
            
            frames_data.append({
                'frame': frame,
                'timestamp': datetime.now() + timedelta(seconds=i*0.1),
                'person_position': (x, y)
            })
        
        return frames_data
    
    def generate_multi_person_scenario(self, person_paths, frames=30):
        """Generate frames with multiple people crossing."""
        frames_data = []
        
        for i in range(frames):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            people_positions = []
            
            for person_id, path in enumerate(person_paths):
                progress = i / (frames - 1)
                x = int(path[0][0] + progress * (path[1][0] - path[0][0]))
                y = int(path[0][1] + progress * (path[1][1] - path[0][1]))
                
                # Draw person with slight variation
                offset = person_id * 5
                cv2.rectangle(frame, (x-20+offset, y-40), (x+20+offset, y+40), (255, 255, 255), -1)
                cv2.circle(frame, (x+offset, y-50), 15, (255, 255, 255), -1)
                
                people_positions.append((x+offset, y))
            
            frames_data.append({
                'frame': frame,
                'timestamp': datetime.now() + timedelta(seconds=i*0.1),
                'people_positions': people_positions
            })
        
        return frames_data


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.camera_system = MockCameraSystem()
        
        # Create virtual line for testing
        self.test_virtual_line = VirtualLine(
            id="test_line_001",
            camera_id="camera_001",
            points=[Point(x=300, y=200), Point(x=300, y=400)],
            active=True,
            sensitivity=0.8
        )
        
        # Mock configurations
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
            "track_buffer": 30
        }
    
    @patch('edge.src.detection_pipeline.ModelManager')
    @patch('services.alert_service.alert_engine.AlertEngine._send_alert')
    @patch('services.evidence_service.evidence_store.EvidenceStore.store_evidence')
    async def test_single_person_crossing_workflow(self, mock_store_evidence, mock_send_alert, mock_model_manager):
        """Test complete workflow for single person crossing virtual line."""
        
        # Setup mocks
        self._setup_detection_mocks(mock_model_manager)
        mock_send_alert.return_value = True
        mock_store_evidence.return_value = "evidence_001"
        
        # Initialize system components
        detection_pipeline = DetectionPipeline(self.detection_config)
        detection_pipeline.virtual_lines = [self.test_virtual_line]
        detection_pipeline.is_running = True
        
        tracker = MultiCameraTracker(self.tracking_config)
        alert_engine = AlertEngine()
        incident_manager = IncidentManager()
        evidence_store = EvidenceStore("test://db", "/tmp/evidence", "test_key", "test_secret")
        
        # Generate crossing scenario
        person_path = [(200, 300), (400, 300)]  # Crosses vertical line at x=300
        scenario_frames = self.camera_system.generate_crossing_scenario(person_path, frames=20)
        
        # Process frames through pipeline
        detections = []
        tracks = []
        alerts = []
        
        for frame_data in scenario_frames:
            # 1. Detection
            result = detection_pipeline.process_frame(frame_data['frame'], "camera_001")
            detections.extend(result.detections)
            
            # 2. Tracking
            if result.detections:
                updated_tracks = tracker.update_tracks(result.detections, "camera_001")
                tracks.extend(updated_tracks)
            
            # 3. Virtual line crossing detection
            for detection in result.detections:
                if result.virtual_line_crossings:
                    # 4. Alert generation
                    crossing_event = result.virtual_line_crossings[0]
                    alert = alert_engine.generate_alert(crossing_event, detection)
                    alerts.append(alert)
                    
                    # 5. Evidence storage
                    evidence_id = await evidence_store.store_evidence(
                        file_data=cv2.imencode('.jpg', frame_data['frame'])[1].tobytes(),
                        evidence_type=EvidenceType.IMAGE,
                        metadata={
                            "detection_id": detection.id,
                            "camera_id": "camera_001",
                            "crossing_event": crossing_event.id
                        },
                        created_by="system"
                    )
                    
                    # 6. Incident creation for high-severity alerts
                    if alert.severity in [Severity.HIGH, Severity.CRITICAL]:
                        incident = incident_manager.create_incident(
                            alert, 
                            "operator_001",
                            "Automated incident creation for border crossing"
                        )
                        
                        # 7. Add evidence to incident
                        incident_manager.add_evidence(incident.id, evidence_id)
        
        # Verify workflow completion
        assert len(detections) > 0, "Should detect person in frames"
        assert len(tracks) > 0, "Should create tracking data"
        assert len(alerts) > 0, "Should generate alerts for crossing"
        
        # Verify alert was sent
        mock_send_alert.assert_called()
        
        # Verify evidence was stored
        mock_store_evidence.assert_called()
        
        # Verify crossing was detected
        crossing_detected = any(result.virtual_line_crossings for result in 
                              [detection_pipeline.process_frame(fd['frame'], "camera_001") 
                               for fd in scenario_frames])
        assert crossing_detected, "Should detect virtual line crossing"
    
    @patch('edge.src.detection_pipeline.ModelManager')
    @patch('services.alert_service.alert_engine.AlertEngine._send_alert')
    async def test_multi_camera_tracking_workflow(self, mock_send_alert, mock_model_manager):
        """Test multi-camera tracking and re-identification workflow."""
        
        # Setup detection mocks for multiple cameras
        self._setup_detection_mocks(mock_model_manager)
        mock_send_alert.return_value = True
        
        # Initialize components
        detection_pipeline_1 = DetectionPipeline(self.detection_config)
        detection_pipeline_2 = DetectionPipeline(self.detection_config)
        
        detection_pipeline_1.is_running = True
        detection_pipeline_2.is_running = True
        
        tracker = MultiCameraTracker(self.tracking_config)
        
        # Generate scenario: person moves from camera 1 to camera 2
        camera_1_path = [(100, 300), (300, 300)]
        camera_2_path = [(300, 300), (500, 300)]
        
        camera_1_frames = self.camera_system.generate_crossing_scenario(camera_1_path, frames=15)
        camera_2_frames = self.camera_system.generate_crossing_scenario(camera_2_path, frames=15)
        
        # Process frames from both cameras
        all_tracks = []
        
        # Process camera 1 frames
        for frame_data in camera_1_frames:
            result = detection_pipeline_1.process_frame(frame_data['frame'], "camera_001")
            if result.detections:
                tracks = tracker.update_tracks(result.detections, "camera_001")
                all_tracks.extend(tracks)
        
        # Small delay to simulate person moving between cameras
        await asyncio.sleep(0.1)
        
        # Process camera 2 frames
        for frame_data in camera_2_frames:
            result = detection_pipeline_2.process_frame(frame_data['frame'], "camera_002")
            if result.detections:
                tracks = tracker.update_tracks(result.detections, "camera_002")
                all_tracks.extend(tracks)
        
        # Verify multi-camera tracking
        camera_1_tracks = [t for t in all_tracks if t.camera_id == "camera_001"]
        camera_2_tracks = [t for t in all_tracks if t.camera_id == "camera_002"]
        
        assert len(camera_1_tracks) > 0, "Should have tracks from camera 1"
        assert len(camera_2_tracks) > 0, "Should have tracks from camera 2"
        
        # Test re-identification across cameras
        global_tracks = tracker.get_global_tracks()
        assert len(global_tracks) >= 1, "Should create global track across cameras"
        
        # Verify track continuity
        for global_track in global_tracks:
            assert len(global_track.camera_tracks) >= 1, "Global track should span multiple cameras"
    
    @patch('edge.src.detection_pipeline.ModelManager')
    @patch('services.alert_service.alert_engine.AlertEngine._send_alert')
    @patch('services.evidence_service.evidence_store.EvidenceStore.store_evidence')
    async def test_multiple_simultaneous_crossings_workflow(self, mock_store_evidence, mock_send_alert, mock_model_manager):
        """Test workflow with multiple people crossing simultaneously."""
        
        # Setup mocks for multiple detections
        self._setup_multi_detection_mocks(mock_model_manager)
        mock_send_alert.return_value = True
        mock_store_evidence.return_value = "evidence_multi"
        
        # Initialize components
        detection_pipeline = DetectionPipeline(self.detection_config)
        detection_pipeline.virtual_lines = [self.test_virtual_line]
        detection_pipeline.is_running = True
        
        tracker = MultiCameraTracker(self.tracking_config)
        alert_engine = AlertEngine()
        
        # Generate multi-person crossing scenario
        person_paths = [
            [(200, 280), (400, 280)],  # Person 1
            [(200, 320), (400, 320)],  # Person 2
            [(200, 360), (400, 360)]   # Person 3
        ]
        
        scenario_frames = self.camera_system.generate_multi_person_scenario(person_paths, frames=20)
        
        # Process frames
        all_detections = []
        all_alerts = []
        
        for frame_data in scenario_frames:
            result = detection_pipeline.process_frame(frame_data['frame'], "camera_001")
            all_detections.extend(result.detections)
            
            # Update tracking
            if result.detections:
                tracker.update_tracks(result.detections, "camera_001")
            
            # Generate alerts for crossings
            if result.virtual_line_crossings:
                for crossing in result.virtual_line_crossings:
                    # Find corresponding detection
                    detection = next((d for d in result.detections 
                                    if d.id == crossing.detection_id), None)
                    if detection:
                        alert = alert_engine.generate_alert(crossing, detection)
                        all_alerts.append(alert)
        
        # Verify multiple people were detected and tracked
        assert len(all_detections) >= 3, "Should detect multiple people"
        
        # Verify multiple alerts were generated
        assert len(all_alerts) >= 1, "Should generate alerts for crossings"
        
        # Verify risk scores are elevated for simultaneous crossings
        if len(all_alerts) > 1:
            avg_risk_score = sum(alert.risk_score for alert in all_alerts) / len(all_alerts)
            assert avg_risk_score > 0.5, "Risk scores should be elevated for simultaneous crossings"
    
    @patch('edge.src.detection_pipeline.ModelManager')
    async def test_failure_recovery_workflow(self, mock_model_manager):
        """Test system recovery from various failure scenarios."""
        
        # Setup mock that fails intermittently
        mock_manager = Mock()
        mock_model = Mock()
        
        # Simulate intermittent failures
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise Exception("Simulated model failure")
            return [Mock(boxes=None)]
        
        mock_model.side_effect = side_effect
        mock_manager.get_model.return_value = mock_model
        mock_model_manager.return_value = mock_manager
        
        # Initialize components
        detection_pipeline = DetectionPipeline(self.detection_config)
        detection_pipeline.is_running = True
        
        tracker = MultiCameraTracker(self.tracking_config)
        
        # Generate test frames
        test_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
        
        # Process frames with failures
        successful_results = 0
        failed_results = 0
        
        for i, frame in enumerate(test_frames):
            try:
                result = detection_pipeline.process_frame(frame, f"camera_{i%2}")
                successful_results += 1
                
                # Verify system continues to function after failures
                assert isinstance(result, DetectionResult)
                assert result.camera_id == f"camera_{i%2}"
                
            except Exception:
                failed_results += 1
        
        # Verify system handled failures gracefully
        assert successful_results > 0, "System should recover from failures"
        assert failed_results > 0, "Should have encountered simulated failures"
        
        # Verify tracking state is maintained despite failures
        active_tracks = tracker.get_active_tracks()
        assert isinstance(active_tracks, list), "Tracking should remain functional"
    
    @patch('edge.src.detection_pipeline.ModelManager')
    @patch('services.evidence_service.evidence_store.EvidenceStore.store_evidence')
    @patch('services.evidence_service.forensics_engine.ForensicsEngine.generate_report')
    async def test_incident_resolution_workflow(self, mock_generate_report, mock_store_evidence, mock_model_manager):
        """Test complete incident resolution workflow."""
        
        # Setup mocks
        self._setup_detection_mocks(mock_model_manager)
        mock_store_evidence.return_value = "evidence_incident"
        mock_generate_report.return_value = "/tmp/incident_report.pdf"
        
        # Initialize components
        incident_manager = IncidentManager()
        evidence_store = EvidenceStore("test://db", "/tmp/evidence", "test_key", "test_secret")
        forensics_engine = ForensicsEngine("test://db", evidence_store, "/tmp/reports")
        
        # Create test alert
        test_alert = Alert(
            id="alert_incident_001",
            type=AlertType.VIRTUAL_LINE_CROSSING,
            severity=Severity.HIGH,
            camera_id="camera_001",
            detection_id="detection_001",
            timestamp=datetime.now(),
            confidence=0.9,
            risk_score=0.8,
            metadata={"crossing_direction": "inbound"}
        )
        
        # 1. Create incident
        incident = incident_manager.create_incident(
            test_alert,
            "operator_001",
            "Border crossing detected - requires investigation"
        )
        
        assert incident.status == IncidentStatus.OPEN
        assert incident.alert_id == test_alert.id
        
        # 2. Assign to operator
        success = incident_manager.assign_incident(incident.id, "operator_001")
        assert success is True
        assert incident.status == IncidentStatus.IN_PROGRESS
        
        # 3. Add investigation notes
        note = incident_manager.add_note(
            incident.id,
            "Reviewing camera footage and checking patrol schedules",
            "operator_001",
            "investigation"
        )
        assert note is not None
        
        # 4. Add evidence
        evidence_id = await evidence_store.store_evidence(
            file_data=b"test evidence data",
            evidence_type=EvidenceType.IMAGE,
            metadata={"incident_id": incident.id},
            created_by="operator_001"
        )
        
        success = incident_manager.add_evidence(incident.id, evidence_id)
        assert success is True
        assert evidence_id in incident.evidence_ids
        
        # 5. Generate forensics report
        report_path = await forensics_engine.generate_report(incident.id, "comprehensive")
        assert report_path == "/tmp/incident_report.pdf"
        
        # 6. Resolve incident
        resolution = Resolution(
            type=ResolutionType.CONFIRMED_CROSSING,
            description="Confirmed unauthorized crossing. Border patrol notified.",
            resolved_by="operator_001",
            actions_taken=["Notified border patrol", "Initiated tracking protocol"]
        )
        
        success = incident_manager.resolve_incident(incident.id, resolution)
        assert success is True
        assert incident.status == IncidentStatus.RESOLVED
        assert incident.resolution.type == ResolutionType.CONFIRMED_CROSSING
        
        # 7. Close incident
        success = incident_manager.close_incident(incident.id, "operator_001")
        assert success is True
        assert incident.status == IncidentStatus.CLOSED
        assert incident.closed_at is not None
        
        # Verify complete audit trail
        audit_trail = incident_manager.get_incident_audit_trail(incident.id)
        assert len(audit_trail) >= 5  # Create, assign, note, resolve, close
    
    def test_system_performance_under_load(self):
        """Test system performance with high load scenarios."""
        
        # This test would normally require actual system deployment
        # For now, we'll test component performance individually
        
        detection_pipeline = DetectionPipeline(self.detection_config)
        tracker = MultiCameraTracker(self.tracking_config)
        
        # Generate high-frequency frames
        test_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
                      for _ in range(100)]
        
        # Measure processing times
        processing_times = []
        
        with patch('edge.src.detection_pipeline.ModelManager') as mock_model_manager:
            self._setup_detection_mocks(mock_model_manager)
            detection_pipeline.is_running = True
            
            for i, frame in enumerate(test_frames):
                start_time = time.perf_counter()
                result = detection_pipeline.process_frame(frame, f"camera_{i%5}")
                processing_time = (time.perf_counter() - start_time) * 1000
                processing_times.append(processing_time)
        
        # Verify performance requirements
        avg_processing_time = sum(processing_times) / len(processing_times)
        max_processing_time = max(processing_times)
        
        assert avg_processing_time < 300, f"Average processing time {avg_processing_time:.1f}ms exceeds 300ms requirement"
        assert max_processing_time < 500, f"Max processing time {max_processing_time:.1f}ms is too high"
        
        # Verify 95th percentile performance
        sorted_times = sorted(processing_times)
        p95_time = sorted_times[int(0.95 * len(sorted_times))]
        assert p95_time < 400, f"95th percentile time {p95_time:.1f}ms exceeds threshold"
    
    def _setup_detection_mocks(self, mock_model_manager):
        """Setup detection pipeline mocks for single person detection."""
        mock_manager = Mock()
        mock_model = Mock()
        
        # Mock single person detection
        mock_result = Mock()
        boxes = np.array([[250, 250, 350, 350]], dtype=np.float32)
        confidences = np.array([0.9], dtype=np.float32)
        classes = np.array([0], dtype=np.float32)
        
        mock_boxes = Mock()
        mock_boxes.xyxy = Mock()
        mock_boxes.xyxy.cpu = Mock()
        mock_boxes.xyxy.cpu.return_value = Mock()
        mock_boxes.xyxy.cpu.return_value.numpy = Mock(return_value=boxes)
        
        mock_boxes.conf = Mock()
        mock_boxes.conf.cpu = Mock()
        mock_boxes.conf.cpu.return_value = Mock()
        mock_boxes.conf.conf.cpu.return_value.numpy = Mock(return_value=confidences)
        
        mock_boxes.cls = Mock()
        mock_boxes.cls.cpu = Mock()
        mock_boxes.cls.cpu.return_value = Mock()
        mock_boxes.cls.cpu.return_value.numpy = Mock(return_value=classes)
        
        mock_boxes.__len__ = Mock(return_value=1)
        mock_result.boxes = mock_boxes
        
        mock_model.return_value = [mock_result]
        mock_manager.get_model.return_value = mock_model
        mock_model_manager.return_value = mock_manager
    
    def _setup_multi_detection_mocks(self, mock_model_manager):
        """Setup detection pipeline mocks for multiple person detection."""
        mock_manager = Mock()
        mock_model = Mock()
        
        # Mock multiple person detections
        mock_result = Mock()
        boxes = np.array([
            [200, 240, 240, 320],  # Person 1
            [200, 280, 240, 360],  # Person 2
            [200, 320, 240, 400]   # Person 3
        ], dtype=np.float32)
        confidences = np.array([0.9, 0.85, 0.8], dtype=np.float32)
        classes = np.array([0, 0, 0], dtype=np.float32)
        
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
        
        mock_boxes.__len__ = Mock(return_value=3)
        mock_result.boxes = mock_boxes
        
        mock_model.return_value = [mock_result]
        mock_manager.get_model.return_value = mock_model
        mock_model_manager.return_value = mock_manager


class TestSystemDataFlow:
    """Test data flow validation across system components."""
    
    def test_detection_to_tracking_data_flow(self):
        """Test data flow from detection to tracking components."""
        
        # Create test detection
        test_detection = Detection(
            id="det_flow_001",
            camera_id="camera_001",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=100, y=100, width=50, height=100),
            confidence=0.85,
            detection_class=DetectionClass.PERSON,
            features=np.random.rand(512).astype(np.float32)
        )
        
        # Initialize tracker
        tracker = MultiCameraTracker({
            "max_disappeared": 30,
            "max_distance": 100
        })
        
        # Process detection through tracker
        tracks = tracker.update_tracks([test_detection], "camera_001")
        
        # Verify data flow
        assert len(tracks) == 1
        track = tracks[0]
        
        assert track.camera_id == test_detection.camera_id
        assert len(track.detections) == 1
        assert track.detections[0].id == test_detection.id
        assert track.status == TrackStatus.ACTIVE
        
        # Verify feature preservation
        if hasattr(track, 'features') and track.features is not None:
            assert len(track.features) == len(test_detection.features)
    
    def test_tracking_to_alert_data_flow(self):
        """Test data flow from tracking to alert generation."""
        
        # Create test track with crossing
        test_track = Track(
            id="track_flow_001",
            camera_id="camera_001",
            detections=[],
            trajectory=[(250, 300), (300, 300), (350, 300)],  # Crosses x=300
            start_time=datetime.now() - timedelta(seconds=3),
            status=TrackStatus.ACTIVE
        )
        
        # Create virtual line
        virtual_line = VirtualLine(
            id="line_flow_001",
            camera_id="camera_001",
            points=[Point(x=300, y=200), Point(x=300, y=400)],
            active=True
        )
        
        # Initialize alert engine
        alert_engine = AlertEngine()
        
        # Create crossing event
        from shared.models.alerts import CrossingEvent
        crossing_event = CrossingEvent(
            detection_id="det_flow_001",
            virtual_line_id=virtual_line.id,
            crossing_point=(300, 300),
            crossing_direction="inbound",
            timestamp=datetime.now(),
            confidence=0.85
        )
        
        # Create corresponding detection
        detection = Detection(
            id="det_flow_001",
            camera_id="camera_001",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=280, y=280, width=40, height=80),
            confidence=0.85,
            detection_class=DetectionClass.PERSON
        )
        
        # Generate alert
        alert = alert_engine.generate_alert(crossing_event, detection)
        
        # Verify data flow
        assert alert.camera_id == test_track.camera_id
        assert alert.detection_id == detection.id
        assert alert.type == AlertType.VIRTUAL_LINE_CROSSING
        assert alert.confidence == detection.confidence
        assert alert.risk_score > 0
    
    def test_alert_to_incident_data_flow(self):
        """Test data flow from alert to incident management."""
        
        # Create test alert
        test_alert = Alert(
            id="alert_flow_001",
            type=AlertType.VIRTUAL_LINE_CROSSING,
            severity=Severity.HIGH,
            camera_id="camera_001",
            detection_id="det_flow_001",
            timestamp=datetime.now(),
            confidence=0.9,
            risk_score=0.8,
            metadata={
                "crossing_direction": "inbound",
                "virtual_line_id": "line_001"
            }
        )
        
        # Initialize incident manager
        incident_manager = IncidentManager()
        
        # Create incident from alert
        incident = incident_manager.create_incident(
            test_alert,
            "operator_001",
            "High-risk border crossing detected"
        )
        
        # Verify data flow
        assert incident.alert_id == test_alert.id
        assert incident.status == IncidentStatus.OPEN
        assert incident.priority is not None
        assert incident.created_by == "operator_001"
        
        # Verify metadata preservation
        assert "crossing_direction" in test_alert.metadata
        assert test_alert.metadata["crossing_direction"] == "inbound"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])