"""
Tests for AlertEngine implementation.
Tests alert generation with various crossing scenarios.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from alert_engine import AlertEngine
from shared.models import Alert, CrossingEvent, Detection
from shared.models.alerts import AlertType, Severity
from shared.models.detection import BoundingBox, DetectionClass


class TestAlertEngine:
    """Test cases for AlertEngine functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.engine = AlertEngine()
        
        # Create test detection
        self.test_detection = Detection(
            id="det_001",
            camera_id="cam_001",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=100, y=100, width=50, height=100),
            confidence=0.85,
            detection_class=DetectionClass.PERSON,
            metadata={'gps_coordinates': (28.6139, 77.2090)}  # Delhi coordinates
        )
        
        # Create test crossing event
        self.test_crossing_event = CrossingEvent(
            detection_id="det_001",
            virtual_line_id="line_001",
            crossing_point=(125, 150),
            crossing_direction="inbound",
            timestamp=datetime.now(),
            confidence=0.85
        )
    
    def test_generate_alert_basic(self):
        """Test basic alert generation from crossing event."""
        alert = self.engine.generate_alert(self.test_crossing_event, self.test_detection)
        
        assert alert is not None
        assert alert.type == AlertType.VIRTUAL_LINE_CROSSING
        assert alert.camera_id == "cam_001"
        assert alert.detection_id == "det_001"
        assert alert.confidence == 0.85
        assert alert.crossing_event == self.test_crossing_event
        assert alert.gps_coordinates == (28.6139, 77.2090)
        assert alert.risk_score > 0
        assert alert.id in self.engine.active_alerts
    
    def test_generate_alert_night_hours(self):
        """Test alert generation during night hours increases risk score."""
        # Create night time crossing event
        night_time = datetime.now().replace(hour=2, minute=30)  # 2:30 AM
        night_crossing = CrossingEvent(
            detection_id="det_002",
            virtual_line_id="line_001",
            crossing_point=(125, 150),
            crossing_direction="inbound",
            timestamp=night_time,
            confidence=0.85
        )
        
        night_detection = Detection(
            id="det_002",
            camera_id="cam_001",
            timestamp=night_time,
            bbox=BoundingBox(x=100, y=100, width=50, height=100),
            confidence=0.85,
            detection_class=DetectionClass.PERSON
        )
        
        alert = self.engine.generate_alert(night_crossing, night_detection)
        
        # Night time should increase risk score
        assert alert.risk_score > 0.6  # Base + night bonus
        assert alert.severity in [Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
    
    def test_calculate_risk_score_high_confidence(self):
        """Test risk score calculation with high confidence detection."""
        high_conf_event = CrossingEvent(
            detection_id="det_003",
            virtual_line_id="line_001",
            crossing_point=(125, 150),
            crossing_direction="inbound",
            timestamp=datetime.now(),
            confidence=0.95  # High confidence
        )
        
        context = {'camera_id': 'cam_001'}
        risk_score = self.engine.calculate_risk_score(high_conf_event, context)
        
        # High confidence should contribute to risk score
        assert risk_score >= 0.4  # Base score from confidence
        assert risk_score <= 1.0
    
    def test_calculate_risk_score_multiple_crossings(self):
        """Test risk score calculation with multiple simultaneous crossings."""
        context = {
            'camera_id': 'cam_001',
            'simultaneous_crossings': 3,
            'group_size': 3
        }
        
        risk_score = self.engine.calculate_risk_score(self.test_crossing_event, context)
        
        # Multiple crossings should increase risk
        assert risk_score > 0.5
    
    def test_calculate_risk_score_restricted_zone(self):
        """Test risk score calculation for restricted zone crossing."""
        context = {
            'camera_id': 'cam_001',
            'restricted_zone': True
        }
        
        risk_score = self.engine.calculate_risk_score(self.test_crossing_event, context)
        
        # Restricted zone should significantly increase risk
        assert risk_score > 0.7
    
    def test_route_alert_handlers(self):
        """Test alert routing to registered handlers."""
        handler_called = []
        
        def test_handler(alert):
            handler_called.append(alert.id)
        
        self.engine.register_alert_handler(test_handler)
        alert = self.engine.generate_alert(self.test_crossing_event, self.test_detection)
        self.engine.route_alert(alert)
        
        assert alert.id in handler_called
    
    def test_acknowledge_alert(self):
        """Test alert acknowledgment functionality."""
        alert = self.engine.generate_alert(self.test_crossing_event, self.test_detection)
        
        success = self.engine.acknowledge_alert(alert.id, "operator_001")
        
        assert success is True
        assert alert.acknowledged is True
        assert alert.acknowledged_by == "operator_001"
        assert alert.acknowledged_at is not None
    
    def test_escalate_alert(self):
        """Test alert escalation functionality."""
        alert = self.engine.generate_alert(self.test_crossing_event, self.test_detection)
        original_severity = alert.severity
        
        success = self.engine.escalate_alert(alert.id, "operator_001", "High risk crossing")
        
        assert success is True
        assert alert.escalated is True
        assert alert.escalated_by == "operator_001"
        assert alert.escalated_at is not None
        assert alert.metadata['escalation_reason'] == "High risk crossing"
        # Severity should be increased
        assert alert.severity.value != original_severity.value or alert.severity == Severity.CRITICAL
    
    def test_get_active_alerts_filtering(self):
        """Test getting active alerts with camera filtering."""
        # Create alerts for different cameras
        alert1 = self.engine.generate_alert(self.test_crossing_event, self.test_detection)
        
        detection2 = Detection(
            id="det_002",
            camera_id="cam_002",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=100, y=100, width=50, height=100),
            confidence=0.75
        )
        crossing2 = CrossingEvent(
            detection_id="det_002",
            virtual_line_id="line_002",
            crossing_point=(125, 150),
            crossing_direction="outbound",
            timestamp=datetime.now(),
            confidence=0.75
        )
        alert2 = self.engine.generate_alert(crossing2, detection2)
        
        # Test filtering by camera
        cam1_alerts = self.engine.get_active_alerts("cam_001")
        cam2_alerts = self.engine.get_active_alerts("cam_002")
        all_alerts = self.engine.get_active_alerts()
        
        assert len(cam1_alerts) == 1
        assert len(cam2_alerts) == 1
        assert len(all_alerts) == 2
        assert cam1_alerts[0].camera_id == "cam_001"
        assert cam2_alerts[0].camera_id == "cam_002"
    
    def test_alert_severity_determination(self):
        """Test alert severity determination based on risk score."""
        # Test different risk scores
        test_cases = [
            (0.9, Severity.CRITICAL),
            (0.7, Severity.HIGH),
            (0.5, Severity.MEDIUM),
            (0.2, Severity.LOW)
        ]
        
        for risk_score, expected_severity in test_cases:
            context = {'test_risk': risk_score}
            # Mock the risk calculation to return specific score
            with patch.object(self.engine, 'calculate_risk_score', return_value=risk_score):
                alert = self.engine.generate_alert(self.test_crossing_event, self.test_detection)
                assert alert.severity == expected_severity
    
    def test_crossing_history_tracking(self):
        """Test crossing history tracking for pattern analysis."""
        camera_id = "cam_001"
        
        # Generate multiple alerts
        for i in range(5):
            detection = Detection(
                id=f"det_{i:03d}",
                camera_id=camera_id,
                timestamp=datetime.now() - timedelta(hours=i),
                bbox=BoundingBox(x=100, y=100, width=50, height=100),
                confidence=0.8
            )
            crossing = CrossingEvent(
                detection_id=f"det_{i:03d}",
                virtual_line_id="line_001",
                crossing_point=(125, 150),
                crossing_direction="inbound",
                timestamp=datetime.now() - timedelta(hours=i),
                confidence=0.8
            )
            self.engine.generate_alert(crossing, detection)
        
        # Check history tracking
        recent_crossings = self.engine._get_recent_crossings(camera_id, hours=24)
        assert len(recent_crossings) == 5
        
        # Check older crossings are filtered out
        old_crossings = self.engine._get_recent_crossings(camera_id, hours=2)
        assert len(old_crossings) <= 3
    
    def test_statistics_generation(self):
        """Test alert engine statistics generation."""
        # Generate some test alerts
        for i in range(3):
            detection = Detection(
                id=f"det_{i:03d}",
                camera_id=f"cam_{i:03d}",
                timestamp=datetime.now(),
                bbox=BoundingBox(x=100, y=100, width=50, height=100),
                confidence=0.8
            )
            crossing = CrossingEvent(
                detection_id=f"det_{i:03d}",
                virtual_line_id="line_001",
                crossing_point=(125, 150),
                crossing_direction="inbound",
                timestamp=datetime.now(),
                confidence=0.8
            )
            self.engine.generate_alert(crossing, detection)
        
        stats = self.engine.get_statistics()
        
        assert stats['total_active_alerts'] == 3
        assert 'severity_breakdown' in stats
        assert stats['total_cameras_with_activity'] == 3
        assert 'alert_handlers_registered' in stats
    
    def test_alert_removal(self):
        """Test alert removal from active alerts."""
        alert = self.engine.generate_alert(self.test_crossing_event, self.test_detection)
        alert_id = alert.id
        
        # Verify alert exists
        assert alert_id in self.engine.active_alerts
        
        # Remove alert
        success = self.engine.remove_alert(alert_id)
        
        assert success is True
        assert alert_id not in self.engine.active_alerts
        
        # Test removing non-existent alert
        success = self.engine.remove_alert("non_existent")
        assert success is False
    
    def test_error_handling(self):
        """Test error handling in alert generation."""
        # Test with invalid detection
        invalid_detection = None
        
        with pytest.raises(Exception):
            self.engine.generate_alert(self.test_crossing_event, invalid_detection)
        
        # Test acknowledging non-existent alert
        success = self.engine.acknowledge_alert("non_existent", "operator_001")
        assert success is False
        
        # Test escalating non-existent alert
        success = self.engine.escalate_alert("non_existent", "operator_001", "test")
        assert success is False