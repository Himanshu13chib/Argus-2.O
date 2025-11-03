"""
Integration tests for the complete alert and incident management system.
Tests end-to-end workflows from alert generation to incident closure.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from alert_engine import AlertEngine
from incident_manager import IncidentManager
from notification_service import NotificationService, EscalationService
from shared.models import Alert, Detection, CrossingEvent
from shared.models.alerts import AlertType, Severity
from shared.models.incidents import IncidentStatus, ResolutionType, Resolution
from shared.models.detection import BoundingBox, DetectionClass


class TestAlertSystemIntegration:
    """Integration tests for the complete alert system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Initialize all components
        self.alert_engine = AlertEngine()
        self.incident_manager = IncidentManager()
        self.notification_service = NotificationService()
        self.escalation_service = EscalationService(self.notification_service)
        
        # Setup alert routing
        self.alert_engine.register_alert_handler(self._handle_alert_for_incident)
        
        # Track created incidents for testing
        self.created_incidents = []
        
        # Create test data
        self.test_detection = Detection(
            id="det_001",
            camera_id="cam_001",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=100, y=100, width=50, height=100),
            confidence=0.85,
            detection_class=DetectionClass.PERSON,
            metadata={'gps_coordinates': (28.6139, 77.2090)}
        )
        
        self.test_crossing_event = CrossingEvent(
            detection_id="det_001",
            virtual_line_id="line_001",
            crossing_point=(125, 150),
            crossing_direction="inbound",
            timestamp=datetime.now(),
            confidence=0.85
        )
    
    def _handle_alert_for_incident(self, alert: Alert):
        """Handler to create incidents from alerts."""
        if alert.severity in [Severity.HIGH, Severity.CRITICAL]:
            incident = self.incident_manager.create_incident(
                alert, 
                "system_operator",
                f"Auto-generated incident for {alert.type.value}"
            )
            self.created_incidents.append(incident)
    
    def test_complete_workflow_normal_alert(self):
        """Test complete workflow for normal severity alert."""
        # Step 1: Generate alert
        alert = self.alert_engine.generate_alert(self.test_crossing_event, self.test_detection)
        
        # Step 2: Route alert (should not create incident for medium/low severity)
        self.alert_engine.route_alert(alert)
        
        # Step 3: Operator acknowledges alert
        success = self.alert_engine.acknowledge_alert(alert.id, "operator_001")
        
        assert success is True
        assert alert.acknowledged is True
        assert len(self.created_incidents) == 0  # No incident for normal alert
    
    def test_complete_workflow_high_severity_alert(self):
        """Test complete workflow for high severity alert with incident creation."""
        # Create high-risk crossing event
        high_risk_event = CrossingEvent(
            detection_id="det_002",
            virtual_line_id="line_001",
            crossing_point=(125, 150),
            crossing_direction="inbound",
            timestamp=datetime.now().replace(hour=2),  # Night time
            confidence=0.95
        )
        
        high_risk_detection = Detection(
            id="det_002",
            camera_id="cam_001",
            timestamp=datetime.now().replace(hour=2),
            bbox=BoundingBox(x=100, y=100, width=50, height=100),
            confidence=0.95,
            gps_coordinates=(28.6139, 77.2090)
        )
        
        # Step 1: Generate high-severity alert
        alert = self.alert_engine.generate_alert(high_risk_event, high_risk_detection)
        
        # Step 2: Route alert (should create incident)
        self.alert_engine.route_alert(alert)
        
        # Step 3: Verify incident was created
        assert len(self.created_incidents) == 1
        incident = self.created_incidents[0]
        assert incident.alert_id == alert.id
        assert incident.status == IncidentStatus.OPEN
        
        # Step 4: Assign incident to operator
        success = self.incident_manager.assign_incident(incident.id, "operator_001")
        assert success is True
        assert incident.status == IncidentStatus.IN_PROGRESS
        
        # Step 5: Add investigation notes
        note = self.incident_manager.add_note(
            incident.id,
            "Reviewing camera footage and cross-referencing with patrol schedules",
            "operator_001",
            "investigation"
        )
        assert note is not None
        
        # Step 6: Add evidence
        success = self.incident_manager.add_evidence(incident.id, "video_evidence_001")
        assert success is True
        assert "video_evidence_001" in incident.evidence_ids
        
        # Step 7: Resolve incident
        resolution = Resolution(
            type=ResolutionType.CONFIRMED_CROSSING,
            description="Confirmed unauthorized crossing. Border patrol dispatched.",
            resolved_by="operator_001",
            actions_taken=["Notified border patrol", "Initiated tracking"]
        )
        success = self.incident_manager.resolve_incident(incident.id, resolution)
        assert success is True
        assert incident.status == IncidentStatus.RESOLVED
        
        # Step 8: Close incident
        success = self.incident_manager.close_incident(incident.id, "operator_001")
        assert success is True
        assert incident.status == IncidentStatus.CLOSED
        assert incident.closed_at is not None
    
    def test_escalation_workflow(self):
        """Test escalation workflow for critical alerts."""
        # Create critical alert
        critical_event = CrossingEvent(
            detection_id="det_003",
            virtual_line_id="line_001",
            crossing_point=(125, 150),
            crossing_direction="inbound",
            timestamp=datetime.now(),
            confidence=0.98
        )
        
        critical_detection = Detection(
            id="det_003",
            camera_id="cam_001",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=100, y=100, width=50, height=100),
            confidence=0.98,
            gps_coordinates=(28.6139, 77.2090)
        )
        
        # Mock notification service for testing
        with patch.object(self.notification_service, 'notify_alert') as mock_notify:
            mock_notify.return_value = {'email': True}
            
            with patch.object(self.escalation_service, '_get_supervisor_contacts') as mock_contacts:
                mock_contacts.return_value = {'email': ['supervisor@example.com']}
                
                # Step 1: Generate critical alert
                alert = self.alert_engine.generate_alert(critical_event, critical_detection)
                
                # Force critical severity for testing
                alert.severity = Severity.CRITICAL
                alert.risk_score = 0.95
                
                # Step 2: Process escalation
                self.escalation_service.process_escalation(alert)
                
                # Step 3: Verify escalation occurred
                history = self.escalation_service.get_escalation_history(alert.id)
                assert len(history) > 0
                
                # Step 4: Route alert (should create incident)
                self.alert_engine.route_alert(alert)
                assert len(self.created_incidents) == 1
                
                incident = self.created_incidents[0]
                
                # Step 5: Escalate incident to supervisor
                success = self.incident_manager.escalate_incident(
                    incident.id,
                    "supervisor_001",
                    "Critical border crossing requires immediate supervisor attention"
                )
                assert success is True
                assert incident.status == IncidentStatus.ESCALATED
                assert incident.supervisor_id == "supervisor_001"
    
    @patch('requests.post')
    def test_notification_workflow(self, mock_post):
        """Test notification workflow for alerts."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Create alert
        alert = self.alert_engine.generate_alert(self.test_crossing_event, self.test_detection)
        
        # Test multi-channel notification
        recipients = ["operator1@example.com", "operator2@example.com"]
        channels = ["email", "webhook"]
        
        with patch.object(self.notification_service, 'send_email') as mock_email:
            mock_email.return_value = True
            
            results = self.notification_service.notify_alert(alert, recipients, channels)
            
            assert results['email'] is True
            assert results['webhook'] is True
            assert mock_email.call_count == 2  # Called for each recipient
            assert mock_post.call_count == 2  # Webhook called for each recipient
    
    def test_multiple_simultaneous_crossings(self):
        """Test handling of multiple simultaneous crossings."""
        alerts = []
        
        # Generate multiple crossing events within short time window
        for i in range(3):
            detection = Detection(
                id=f"det_{i:03d}",
                camera_id="cam_001",
                timestamp=datetime.now(),
                bbox=BoundingBox(x=100 + i*10, y=100, width=50, height=100),
                confidence=0.8 + i*0.05,
                gps_coordinates=(28.6139 + i*0.001, 77.2090)
            )
            
            crossing = CrossingEvent(
                detection_id=f"det_{i:03d}",
                virtual_line_id="line_001",
                crossing_point=(125 + i*10, 150),
                crossing_direction="inbound",
                timestamp=datetime.now(),
                confidence=0.8 + i*0.05
            )
            
            alert = self.alert_engine.generate_alert(crossing, detection)
            alerts.append(alert)
        
        # Verify risk scores are elevated due to simultaneous crossings
        for alert in alerts[1:]:  # Skip first alert
            assert alert.risk_score > 0.5  # Should be elevated
        
        # Route all alerts
        for alert in alerts:
            self.alert_engine.route_alert(alert)
        
        # Should create incidents for high-risk alerts
        assert len(self.created_incidents) >= 1
    
    def test_false_positive_workflow(self):
        """Test workflow for false positive resolution."""
        # Generate alert and create incident
        alert = self.alert_engine.generate_alert(self.test_crossing_event, self.test_detection)
        alert.severity = Severity.HIGH  # Force high severity
        self.alert_engine.route_alert(alert)
        
        assert len(self.created_incidents) == 1
        incident = self.created_incidents[0]
        
        # Assign to operator
        self.incident_manager.assign_incident(incident.id, "operator_001")
        
        # Add investigation note
        self.incident_manager.add_note(
            incident.id,
            "Reviewed footage - appears to be authorized maintenance personnel",
            "operator_001",
            "investigation"
        )
        
        # Resolve as false positive
        resolution = Resolution(
            type=ResolutionType.FALSE_POSITIVE,
            description="False positive - authorized maintenance personnel with proper clearance",
            resolved_by="operator_001",
            actions_taken=["Verified personnel authorization", "Updated patrol schedule"]
        )
        
        success = self.incident_manager.resolve_incident(incident.id, resolution)
        assert success is True
        assert incident.resolution.type == ResolutionType.FALSE_POSITIVE
        
        # Close incident
        success = self.incident_manager.close_incident(incident.id, "operator_001")
        assert success is True
    
    def test_overdue_incident_handling(self):
        """Test handling of overdue incidents."""
        # Create old incident
        alert = self.alert_engine.generate_alert(self.test_crossing_event, self.test_detection)
        alert.severity = Severity.HIGH
        self.alert_engine.route_alert(alert)
        
        incident = self.created_incidents[0]
        # Simulate old incident
        incident.created_at = datetime.now() - timedelta(hours=25)
        
        # Get overdue incidents
        overdue = self.incident_manager.get_overdue_incidents(sla_hours=24)
        assert len(overdue) == 1
        assert overdue[0].id == incident.id
        
        # Test SLA-based overdue detection
        incident.priority = incident.priority.HIGH
        overdue_high = self.incident_manager.get_overdue_incidents()
        assert len(overdue_high) == 1  # Should use priority-specific SLA
    
    def test_system_statistics_and_monitoring(self):
        """Test system statistics and monitoring capabilities."""
        # Generate various alerts and incidents
        for i in range(5):
            detection = Detection(
                id=f"det_{i:03d}",
                camera_id=f"cam_{i%2:03d}",  # Alternate between 2 cameras
                timestamp=datetime.now() - timedelta(hours=i),
                bbox=BoundingBox(x=100, y=100, width=50, height=100),
                confidence=0.7 + i*0.05
            )
            
            crossing = CrossingEvent(
                detection_id=f"det_{i:03d}",
                virtual_line_id="line_001",
                crossing_point=(125, 150),
                crossing_direction="inbound" if i % 2 == 0 else "outbound",
                timestamp=datetime.now() - timedelta(hours=i),
                confidence=0.7 + i*0.05
            )
            
            alert = self.alert_engine.generate_alert(crossing, detection)
            if i < 2:  # Make first 2 high severity
                alert.severity = Severity.HIGH
                self.alert_engine.route_alert(alert)
        
        # Get alert engine statistics
        alert_stats = self.alert_engine.get_statistics()
        assert alert_stats['total_active_alerts'] == 5
        assert alert_stats['total_cameras_with_activity'] == 2
        
        # Get incident manager statistics
        incident_stats = self.incident_manager.get_incident_statistics(time_range_hours=24)
        assert incident_stats['total_incidents'] == 2  # Only high severity created incidents
        assert 'status_breakdown' in incident_stats
        assert 'priority_breakdown' in incident_stats
    
    def test_error_recovery_and_resilience(self):
        """Test system behavior under error conditions."""
        # Test alert generation with invalid data
        invalid_detection = Detection(
            id="",  # Invalid empty ID
            camera_id="cam_001",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=100, y=100, width=50, height=100),
            confidence=0.8
        )
        
        # Should handle gracefully
        try:
            alert = self.alert_engine.generate_alert(self.test_crossing_event, invalid_detection)
            # If no exception, verify alert was created with fallback values
            assert alert is not None
        except Exception:
            # Exception is acceptable for invalid data
            pass
        
        # Test incident operations with non-existent IDs
        success = self.incident_manager.assign_incident("non_existent", "operator_001")
        assert success is False
        
        # Test notification with invalid configuration
        invalid_service = NotificationService({})  # Empty config
        success = invalid_service.send_sms("+1234567890", "test")
        assert success is False  # Should fail gracefully
    
    def test_concurrent_alert_processing(self):
        """Test system behavior with concurrent alert processing."""
        import threading
        import time
        
        alerts_created = []
        
        def create_alert(index):
            detection = Detection(
                id=f"det_concurrent_{index}",
                camera_id="cam_001",
                timestamp=datetime.now(),
                bbox=BoundingBox(x=100, y=100, width=50, height=100),
                confidence=0.8
            )
            
            crossing = CrossingEvent(
                detection_id=f"det_concurrent_{index}",
                virtual_line_id="line_001",
                crossing_point=(125, 150),
                crossing_direction="inbound",
                timestamp=datetime.now(),
                confidence=0.8
            )
            
            alert = self.alert_engine.generate_alert(crossing, detection)
            alerts_created.append(alert)
        
        # Create multiple threads to simulate concurrent processing
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_alert, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all alerts were created successfully
        assert len(alerts_created) == 5
        assert len(self.alert_engine.active_alerts) >= 5
        
        # Verify no data corruption occurred
        for alert in alerts_created:
            assert alert.id in self.alert_engine.active_alerts
            assert alert.camera_id == "cam_001"