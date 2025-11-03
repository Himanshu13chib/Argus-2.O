"""
Tests for IncidentManager implementation.
Tests incident management workflows from creation to closure.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from incident_manager import IncidentManager
from shared.models import Alert, Incident, Note, Resolution
from shared.models.alerts import AlertType, Severity
from shared.models.incidents import IncidentStatus, IncidentPriority, ResolutionType


class TestIncidentManager:
    """Test cases for IncidentManager functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.manager = IncidentManager()
        
        # Create test alert
        self.test_alert = Alert(
            id="alert_001",
            type=AlertType.VIRTUAL_LINE_CROSSING,
            severity=Severity.HIGH,
            camera_id="cam_001",
            detection_id="det_001",
            timestamp=datetime.now(),
            confidence=0.85,
            risk_score=0.75,
            gps_coordinates=(28.6139, 77.2090)
        )
    
    def test_create_incident_basic(self):
        """Test basic incident creation from alert."""
        incident = self.manager.create_incident(
            self.test_alert, 
            "operator_001",
            "Test Border Crossing",
            "Test incident description"
        )
        
        assert incident is not None
        assert incident.alert_id == "alert_001"
        assert incident.operator_id == "operator_001"
        assert incident.title == "Test Border Crossing"
        assert incident.description == "Test incident description"
        assert incident.status == IncidentStatus.OPEN
        assert incident.gps_coordinates == (28.6139, 77.2090)
        assert incident.camera_ids == ["cam_001"]
        assert len(incident.notes) == 1  # Initial creation note
        assert incident.id in self.manager.incidents
    
    def test_create_incident_auto_title(self):
        """Test incident creation with auto-generated title."""
        incident = self.manager.create_incident(self.test_alert, "operator_001")
        
        assert "Border Crossing Alert" in incident.title
        assert "Virtual Line Crossing" in incident.title
    
    def test_create_incident_priority_mapping(self):
        """Test incident priority mapping from alert severity."""
        test_cases = [
            (Severity.CRITICAL, IncidentPriority.CRITICAL),
            (Severity.HIGH, IncidentPriority.HIGH),
            (Severity.MEDIUM, IncidentPriority.MEDIUM),
            (Severity.LOW, IncidentPriority.LOW)
        ]
        
        for alert_severity, expected_priority in test_cases:
            alert = Alert(
                type=AlertType.VIRTUAL_LINE_CROSSING,
                severity=alert_severity,
                camera_id="cam_001",
                confidence=0.8,
                risk_score=0.6
            )
            
            incident = self.manager.create_incident(alert, "operator_001")
            assert incident.priority == expected_priority
    
    def test_update_incident(self):
        """Test incident updating functionality."""
        incident = self.manager.create_incident(self.test_alert, "operator_001")
        incident_id = incident.id
        
        updates = {
            'title': 'Updated Title',
            'description': 'Updated description',
            'status': IncidentStatus.IN_PROGRESS,
            'updated_by': 'operator_002'
        }
        
        updated_incident = self.manager.update_incident(incident_id, updates)
        
        assert updated_incident is not None
        assert updated_incident.title == 'Updated Title'
        assert updated_incident.description == 'Updated description'
        assert updated_incident.status == IncidentStatus.IN_PROGRESS
        assert updated_incident.updated_at > incident.created_at
        
        # Check status change note was added
        status_notes = [note for note in updated_incident.notes if note.note_type == "action"]
        assert len(status_notes) >= 1
    
    def test_add_note(self):
        """Test adding notes to incidents."""
        incident = self.manager.create_incident(self.test_alert, "operator_001")
        
        note = self.manager.add_note(
            incident.id,
            "This is a test note",
            "operator_002",
            "investigation"
        )
        
        assert note is not None
        assert note.content == "This is a test note"
        assert note.author_id == "operator_002"
        assert note.note_type == "investigation"
        assert note in incident.notes
    
    def test_add_evidence(self):
        """Test linking evidence to incidents."""
        incident = self.manager.create_incident(self.test_alert, "operator_001")
        
        success = self.manager.add_evidence(incident.id, "evidence_001")
        
        assert success is True
        assert "evidence_001" in incident.evidence_ids
        
        # Check evidence note was added
        evidence_notes = [note for note in incident.notes if note.note_type == "evidence"]
        assert len(evidence_notes) >= 1
    
    def test_assign_incident(self):
        """Test incident assignment to operators."""
        incident = self.manager.create_incident(self.test_alert, "operator_001")
        
        success = self.manager.assign_incident(incident.id, "operator_002")
        
        assert success is True
        assert incident.assigned_to == "operator_002"
        assert incident.status == IncidentStatus.IN_PROGRESS
        assert incident.id in self.manager.operator_assignments["operator_002"]
        
        # Check assignment note was added
        assignment_notes = [note for note in incident.notes if note.note_type == "action"]
        assert any("assigned to operator" in note.content.lower() for note in assignment_notes)
    
    def test_escalate_incident(self):
        """Test incident escalation to supervisor."""
        incident = self.manager.create_incident(self.test_alert, "operator_001")
        
        success = self.manager.escalate_incident(
            incident.id,
            "supervisor_001",
            "High risk crossing requires immediate attention"
        )
        
        assert success is True
        assert incident.supervisor_id == "supervisor_001"
        assert incident.supervisor_notified is True
        assert incident.escalation_reason == "High risk crossing requires immediate attention"
        assert incident.status == IncidentStatus.ESCALATED
        assert incident.priority == IncidentPriority.HIGH
        
        # Check escalation note was added
        escalation_notes = [note for note in incident.notes if note.note_type == "escalation"]
        assert len(escalation_notes) >= 1
    
    def test_resolve_incident(self):
        """Test incident resolution."""
        incident = self.manager.create_incident(self.test_alert, "operator_001")
        
        resolution = Resolution(
            type=ResolutionType.CONFIRMED_CROSSING,
            description="Confirmed unauthorized border crossing",
            resolved_by="operator_001",
            actions_taken=["Notified border patrol", "Initiated pursuit"]
        )
        
        success = self.manager.resolve_incident(incident.id, resolution)
        
        assert success is True
        assert incident.resolution == resolution
        assert incident.status == IncidentStatus.RESOLVED
        
        # Check resolution note was added
        resolution_notes = [note for note in incident.notes if note.note_type == "resolution"]
        assert len(resolution_notes) >= 1
    
    def test_close_incident(self):
        """Test incident closure."""
        incident = self.manager.create_incident(self.test_alert, "operator_001")
        
        # First resolve the incident
        resolution = Resolution(
            type=ResolutionType.FALSE_POSITIVE,
            description="False positive - authorized personnel",
            resolved_by="operator_001"
        )
        self.manager.resolve_incident(incident.id, resolution)
        
        # Then close it
        success = self.manager.close_incident(incident.id, "operator_001")
        
        assert success is True
        assert incident.status == IncidentStatus.CLOSED
        assert incident.closed_at is not None
        
        # Check closure note was added
        closure_notes = [note for note in incident.notes if "closed by operator" in note.content.lower()]
        assert len(closure_notes) >= 1
    
    def test_close_incident_not_resolved(self):
        """Test that unresolved incidents cannot be closed."""
        incident = self.manager.create_incident(self.test_alert, "operator_001")
        
        success = self.manager.close_incident(incident.id, "operator_001")
        
        assert success is False
        assert incident.status != IncidentStatus.CLOSED
    
    def test_get_incidents_by_status(self):
        """Test filtering incidents by status."""
        # Create incidents with different statuses
        incident1 = self.manager.create_incident(self.test_alert, "operator_001")
        
        alert2 = Alert(
            type=AlertType.LOITERING_DETECTED,
            severity=Severity.MEDIUM,
            camera_id="cam_002",
            confidence=0.7,
            risk_score=0.5
        )
        incident2 = self.manager.create_incident(alert2, "operator_002")
        self.manager.assign_incident(incident2.id, "operator_002")  # Changes to IN_PROGRESS
        
        open_incidents = self.manager.get_incidents_by_status(IncidentStatus.OPEN)
        in_progress_incidents = self.manager.get_incidents_by_status(IncidentStatus.IN_PROGRESS)
        
        assert len(open_incidents) == 1
        assert len(in_progress_incidents) == 1
        assert open_incidents[0].id == incident1.id
        assert in_progress_incidents[0].id == incident2.id
    
    def test_get_incidents_by_operator(self):
        """Test getting incidents assigned to specific operator."""
        incident1 = self.manager.create_incident(self.test_alert, "operator_001")
        self.manager.assign_incident(incident1.id, "operator_002")
        
        alert2 = Alert(
            type=AlertType.TAMPER_DETECTED,
            severity=Severity.HIGH,
            camera_id="cam_003",
            confidence=0.9,
            risk_score=0.8
        )
        incident2 = self.manager.create_incident(alert2, "operator_001")
        self.manager.assign_incident(incident2.id, "operator_002")
        
        operator2_incidents = self.manager.get_incidents_by_operator("operator_002")
        
        assert len(operator2_incidents) == 2
        assert all(inc.assigned_to == "operator_002" for inc in operator2_incidents)
    
    def test_get_overdue_incidents(self):
        """Test getting overdue incidents based on SLA."""
        # Create old incident
        old_incident = self.manager.create_incident(self.test_alert, "operator_001")
        old_incident.created_at = datetime.now() - timedelta(hours=25)  # 25 hours ago
        
        # Create recent incident
        recent_alert = Alert(
            type=AlertType.VIRTUAL_LINE_CROSSING,
            severity=Severity.LOW,
            camera_id="cam_002",
            confidence=0.6,
            risk_score=0.3
        )
        recent_incident = self.manager.create_incident(recent_alert, "operator_001")
        
        overdue_incidents = self.manager.get_overdue_incidents(sla_hours=24)
        
        assert len(overdue_incidents) == 1
        assert overdue_incidents[0].id == old_incident.id
    
    def test_search_incidents(self):
        """Test incident search functionality."""
        # Create incidents with different content
        incident1 = self.manager.create_incident(
            self.test_alert, 
            "operator_001",
            "Border Crossing Alert",
            "Suspicious person detected crossing border"
        )
        
        alert2 = Alert(
            type=AlertType.LOITERING_DETECTED,
            severity=Severity.MEDIUM,
            camera_id="cam_002",
            confidence=0.7,
            risk_score=0.5
        )
        incident2 = self.manager.create_incident(
            alert2,
            "operator_002", 
            "Loitering Detection",
            "Person loitering near restricted area"
        )
        
        # Add notes
        self.manager.add_note(incident1.id, "Border patrol notified", "operator_001")
        self.manager.add_note(incident2.id, "Security team dispatched", "operator_002")
        
        # Test search by title
        border_results = self.manager.search_incidents("border")
        assert len(border_results) == 1
        assert border_results[0].id == incident1.id
        
        # Test search by description
        loitering_results = self.manager.search_incidents("loitering")
        assert len(loitering_results) == 1
        assert loitering_results[0].id == incident2.id
        
        # Test search by notes
        patrol_results = self.manager.search_incidents("patrol")
        assert len(patrol_results) == 1
        assert patrol_results[0].id == incident1.id
    
    def test_search_incidents_with_filters(self):
        """Test incident search with filters."""
        incident1 = self.manager.create_incident(self.test_alert, "operator_001")
        self.manager.assign_incident(incident1.id, "operator_002")
        
        alert2 = Alert(
            type=AlertType.LOITERING_DETECTED,
            severity=Severity.LOW,
            camera_id="cam_002",
            confidence=0.6,
            risk_score=0.3
        )
        incident2 = self.manager.create_incident(alert2, "operator_003")
        
        # Search with filters
        filters = {
            'status': IncidentStatus.IN_PROGRESS.value,
            'camera_id': 'cam_001'
        }
        
        results = self.manager.search_incidents("", filters)
        
        assert len(results) == 1
        assert results[0].id == incident1.id
    
    def test_get_incident_statistics(self):
        """Test incident statistics generation."""
        # Create various incidents
        incident1 = self.manager.create_incident(self.test_alert, "operator_001")
        
        alert2 = Alert(
            type=AlertType.LOITERING_DETECTED,
            severity=Severity.MEDIUM,
            camera_id="cam_002",
            confidence=0.7,
            risk_score=0.5
        )
        incident2 = self.manager.create_incident(alert2, "operator_002")
        
        # Resolve one incident
        resolution = Resolution(
            type=ResolutionType.FALSE_POSITIVE,
            description="False alarm",
            resolved_by="operator_001"
        )
        self.manager.resolve_incident(incident1.id, resolution)
        self.manager.close_incident(incident1.id, "operator_001")
        
        stats = self.manager.get_incident_statistics(time_range_hours=24)
        
        assert stats['total_incidents'] == 2
        assert stats['resolved_incidents'] == 1
        assert stats['resolution_rate'] == 0.5
        assert 'status_breakdown' in stats
        assert 'priority_breakdown' in stats
        assert 'resolution_breakdown' in stats
        assert stats['avg_resolution_time_hours'] > 0
    
    def test_auto_assignment_rules(self):
        """Test automatic assignment rules."""
        # Add auto-assignment rule
        rule = {
            'name': 'high_priority_rule',
            'conditions': {
                'priority': [IncidentPriority.HIGH.value, IncidentPriority.CRITICAL.value]
            },
            'operator_id': 'senior_operator_001'
        }
        self.manager.add_auto_assignment_rule(rule)
        
        # Create high priority incident
        high_priority_alert = Alert(
            type=AlertType.VIRTUAL_LINE_CROSSING,
            severity=Severity.HIGH,
            camera_id="cam_001",
            confidence=0.9,
            risk_score=0.8
        )
        
        incident = self.manager.create_incident(high_priority_alert, "operator_001")
        
        # Should be auto-assigned
        assert incident.assigned_to == 'senior_operator_001'
        assert incident.status == IncidentStatus.IN_PROGRESS
    
    def test_error_handling(self):
        """Test error handling in incident management."""
        # Test updating non-existent incident
        result = self.manager.update_incident("non_existent", {'title': 'test'})
        assert result is None
        
        # Test adding note to non-existent incident
        note = self.manager.add_note("non_existent", "test note", "operator_001")
        assert note is None
        
        # Test adding evidence to non-existent incident
        success = self.manager.add_evidence("non_existent", "evidence_001")
        assert success is False
        
        # Test assigning non-existent incident
        success = self.manager.assign_incident("non_existent", "operator_001")
        assert success is False