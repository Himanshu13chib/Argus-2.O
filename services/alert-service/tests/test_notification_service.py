"""
Tests for NotificationService and EscalationService implementations.
Tests escalation workflows and notification delivery.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import smtplib
import requests

from notification_service import NotificationService, EscalationService
from shared.models import Alert
from shared.models.alerts import AlertType, Severity, CrossingEvent


class TestNotificationService:
    """Test cases for NotificationService functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'smtp_server': 'test.smtp.com',
            'smtp_port': 587,
            'smtp_username': 'test@example.com',
            'smtp_password': 'password',
            'from_email': 'alerts@projectargus.com',
            'sms_api_url': 'https://api.sms.com/send',
            'sms_api_key': 'test_sms_key',
            'push_api_url': 'https://api.push.com/send',
            'push_api_key': 'test_push_key'
        }
        self.service = NotificationService(self.config)
        
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
            gps_coordinates=(28.6139, 77.2090),
            crossing_event=CrossingEvent(
                detection_id="det_001",
                virtual_line_id="line_001",
                crossing_point=(125, 150),
                crossing_direction="inbound",
                timestamp=datetime.now(),
                confidence=0.85
            )
        )
    
    @patch('smtplib.SMTP')
    def test_send_email_success(self, mock_smtp):
        """Test successful email sending."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        success = self.service.send_email(
            "test@example.com",
            "Test Alert",
            "This is a test alert message"
        )
        
        assert success is True
        mock_smtp.assert_called_once_with('test.smtp.com', 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with('test@example.com', 'password')
        mock_server.send_message.assert_called_once()
    
    @patch('smtplib.SMTP')
    def test_send_email_failure(self, mock_smtp):
        """Test email sending failure handling."""
        mock_smtp.side_effect = Exception("SMTP connection failed")
        
        success = self.service.send_email(
            "test@example.com",
            "Test Alert",
            "This is a test alert message"
        )
        
        assert success is False
    
    @patch('requests.post')
    def test_send_sms_success(self, mock_post):
        """Test successful SMS sending."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        success = self.service.send_sms("+91-9876543210", "Test SMS alert")
        
        assert success is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['to'] == "+91-9876543210"
        assert call_args[1]['json']['message'] == "Test SMS alert"
        assert call_args[1]['json']['api_key'] == "test_sms_key"
    
    @patch('requests.post')
    def test_send_sms_failure(self, mock_post):
        """Test SMS sending failure handling."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response
        
        success = self.service.send_sms("+91-9876543210", "Test SMS alert")
        
        assert success is False
    
    @patch('requests.post')
    def test_send_push_notification_success(self, mock_post):
        """Test successful push notification sending."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        success = self.service.send_push_notification(
            "user_001",
            "Border Alert",
            "Crossing detected",
            {"alert_id": "alert_001"}
        )
        
        assert success is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['user_id'] == "user_001"
        assert call_args[1]['json']['title'] == "Border Alert"
        assert call_args[1]['json']['data']['alert_id'] == "alert_001"
    
    @patch('requests.post')
    def test_send_webhook_success(self, mock_post):
        """Test successful webhook sending."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        payload = {"alert_id": "alert_001", "status": "triggered"}
        success = self.service.send_webhook("https://external.api.com/webhook", payload)
        
        assert success is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json'] == payload
        assert call_args[1]['headers']['Content-Type'] == 'application/json'
    
    @patch('requests.post')
    def test_send_webhook_failure(self, mock_post):
        """Test webhook sending failure handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        payload = {"alert_id": "alert_001"}
        success = self.service.send_webhook("https://external.api.com/webhook", payload)
        
        assert success is False
    
    @patch.object(NotificationService, 'send_email')
    @patch.object(NotificationService, 'send_sms')
    @patch.object(NotificationService, 'send_push_notification')
    def test_notify_alert_multi_channel(self, mock_push, mock_sms, mock_email):
        """Test multi-channel alert notification."""
        mock_email.return_value = True
        mock_sms.return_value = True
        mock_push.return_value = True
        
        recipients = ["operator1@example.com", "operator2@example.com"]
        channels = ["email", "sms", "push"]
        
        results = self.service.notify_alert(self.test_alert, recipients, channels)
        
        assert results['email'] is True
        assert results['sms'] is True
        assert results['push'] is True
        
        # Verify each channel was called for each recipient
        assert mock_email.call_count == 2
        assert mock_sms.call_count == 2
        assert mock_push.call_count == 2
    
    @patch.object(NotificationService, 'send_email')
    def test_notify_alert_partial_failure(self, mock_email):
        """Test alert notification with partial failures."""
        # First call succeeds, second fails
        mock_email.side_effect = [True, False]
        
        recipients = ["success@example.com", "fail@example.com"]
        channels = ["email"]
        
        results = self.service.notify_alert(self.test_alert, recipients, channels)
        
        assert results['email'] is False  # Should be False if any recipient fails
        assert mock_email.call_count == 2
    
    def test_format_alert_message(self):
        """Test alert message formatting."""
        message = self.service._format_alert_message(self.test_alert)
        
        assert "BORDER SECURITY ALERT" in message
        assert "Virtual Line Crossing" in message
        assert "HIGH" in message
        assert "cam_001" in message
        assert "0.85" in message  # confidence
        assert "0.75" in message  # risk score
        assert "28.6139" in message  # GPS coordinates
        assert "inbound" in message  # crossing direction
        assert "alert_001" in message  # alert ID
    
    def test_format_alert_html(self):
        """Test HTML alert formatting."""
        html = self.service._format_alert_html(self.test_alert)
        
        assert "<html>" in html
        assert "Border Security Alert" in html
        assert "Virtual Line Crossing" in html
        assert "#fd7e14" in html  # HIGH severity color
        assert "cam_001" in html
        assert "28.6139" in html
        assert "inbound" in html
    
    def test_format_alert_sms(self):
        """Test SMS alert formatting (short format)."""
        sms = self.service._format_alert_sms(self.test_alert)
        
        assert len(sms) <= 160  # SMS length limit
        assert "BORDER ALERT" in sms
        assert "Virtual Line Crossing" in sms
        assert "cam_001" in sms
        assert "HIGH" in sms
        assert "alert_001"[:8] in sms  # Shortened alert ID
    
    def test_get_notification_status(self):
        """Test notification status retrieval."""
        # Test non-existent notification
        status = self.service.get_notification_status("non_existent")
        assert status['status'] == 'not_found'
        
        # Create a notification history entry
        notification_id = "notif_001"
        self.service.notification_history[notification_id] = {
            'alert_id': 'alert_001',
            'status': 'completed',
            'timestamp': datetime.now()
        }
        
        status = self.service.get_notification_status(notification_id)
        assert status['status'] == 'completed'
        assert status['alert_id'] == 'alert_001'


class TestEscalationService:
    """Test cases for EscalationService functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.notification_service = Mock(spec=NotificationService)
        self.escalation_service = EscalationService(self.notification_service)
        
        # Create test alert
        self.test_alert = Alert(
            id="alert_001",
            type=AlertType.VIRTUAL_LINE_CROSSING,
            severity=Severity.CRITICAL,
            camera_id="cam_001",
            detection_id="det_001",
            timestamp=datetime.now(),
            confidence=0.95,
            risk_score=0.9,
            metadata={'simultaneous_crossings': 3}
        )
    
    def test_register_escalation_rule(self):
        """Test escalation rule registration."""
        def test_condition(alert):
            return alert.severity == Severity.CRITICAL
        
        def test_action(alert):
            pass
        
        self.escalation_service.register_escalation_rule(
            "test_rule",
            test_condition,
            test_action
        )
        
        assert "test_rule" in self.escalation_service.escalation_rules
        assert self.escalation_service.escalation_rules["test_rule"]["condition"] == test_condition
        assert self.escalation_service.escalation_rules["test_rule"]["action"] == test_action
    
    def test_process_escalation_critical_alert(self):
        """Test escalation processing for critical alerts."""
        # Mock the notify_supervisor method
        with patch.object(self.escalation_service, 'notify_supervisor') as mock_notify:
            mock_notify.return_value = True
            
            self.escalation_service.process_escalation(self.test_alert)
            
            # Critical alert should trigger escalation
            assert len(self.escalation_service.escalation_history[self.test_alert.id]) > 0
            escalation = self.escalation_service.escalation_history[self.test_alert.id][0]
            assert escalation['rule'] == 'critical_auto_escalation'
            assert escalation['alert_severity'] == 'critical'
    
    def test_process_escalation_high_risk_score(self):
        """Test escalation processing for high risk score alerts."""
        high_risk_alert = Alert(
            id="alert_002",
            type=AlertType.VIRTUAL_LINE_CROSSING,
            severity=Severity.MEDIUM,
            camera_id="cam_001",
            confidence=0.9,
            risk_score=0.85  # High risk score
        )
        
        with patch.object(self.escalation_service, 'notify_supervisor') as mock_notify:
            with patch.object(self.escalation_service, '_get_duty_supervisor', return_value='duty_officer'):
                mock_notify.return_value = True
                
                self.escalation_service.process_escalation(high_risk_alert)
                
                # High risk should trigger escalation
                assert len(self.escalation_service.escalation_history[high_risk_alert.id]) > 0
                escalation = self.escalation_service.escalation_history[high_risk_alert.id][0]
                assert escalation['rule'] == 'high_risk_escalation'
    
    def test_process_escalation_multiple_crossings(self):
        """Test escalation processing for multiple crossings."""
        multiple_crossing_alert = Alert(
            id="alert_003",
            type=AlertType.MULTIPLE_CROSSINGS,
            severity=Severity.HIGH,
            camera_id="cam_001",
            confidence=0.8,
            risk_score=0.7,
            metadata={'simultaneous_crossings': 4}
        )
        
        with patch.object(self.escalation_service, '_notify_security_team') as mock_notify_team:
            self.escalation_service.process_escalation(multiple_crossing_alert)
            
            # Multiple crossings should trigger security team notification
            mock_notify_team.assert_called_once_with(multiple_crossing_alert)
            assert len(self.escalation_service.escalation_history[multiple_crossing_alert.id]) > 0
    
    def test_notify_supervisor_success(self):
        """Test successful supervisor notification."""
        self.notification_service.notify_alert.return_value = {'email': True}
        
        with patch.object(self.escalation_service, '_get_supervisor_contacts') as mock_contacts:
            mock_contacts.return_value = {'email': ['supervisor@example.com']}
            
            success = self.escalation_service.notify_supervisor(self.test_alert, "supervisor_001")
            
            assert success is True
            self.notification_service.notify_alert.assert_called_once()
            call_args = self.notification_service.notify_alert.call_args
            assert call_args[0][0] == self.test_alert  # alert
            assert call_args[0][1] == ['supervisor@example.com']  # recipients
            assert 'email' in call_args[0][2]  # channels
    
    def test_notify_supervisor_critical_multi_channel(self):
        """Test supervisor notification uses multiple channels for critical alerts."""
        critical_alert = Alert(
            severity='critical',  # String value for this test
            id="alert_critical",
            type=AlertType.VIRTUAL_LINE_CROSSING,
            camera_id="cam_001",
            confidence=0.95,
            risk_score=0.95
        )
        
        self.notification_service.notify_alert.return_value = {'email': True, 'sms': True, 'push': True}
        
        with patch.object(self.escalation_service, '_get_supervisor_contacts') as mock_contacts:
            mock_contacts.return_value = {'email': ['supervisor@example.com']}
            
            success = self.escalation_service.notify_supervisor(critical_alert, "supervisor_001")
            
            assert success is True
            call_args = self.notification_service.notify_alert.call_args
            channels = call_args[0][2]
            assert 'email' in channels
            assert 'sms' in channels
            assert 'push' in channels
    
    def test_notify_supervisor_no_contacts(self):
        """Test supervisor notification when no contact info available."""
        with patch.object(self.escalation_service, '_get_supervisor_contacts') as mock_contacts:
            mock_contacts.return_value = {}
            
            success = self.escalation_service.notify_supervisor(self.test_alert, "supervisor_001")
            
            assert success is False
            self.notification_service.notify_alert.assert_not_called()
    
    def test_get_escalation_history(self):
        """Test escalation history retrieval."""
        # Add some escalation history
        alert_id = "alert_001"
        self.escalation_service.escalation_history[alert_id] = [
            {
                'rule': 'critical_auto_escalation',
                'timestamp': datetime.now(),
                'alert_severity': 'critical',
                'alert_risk_score': 0.9
            }
        ]
        
        history = self.escalation_service.get_escalation_history(alert_id)
        
        assert len(history) == 1
        assert history[0]['rule'] == 'critical_auto_escalation'
        assert history[0]['alert_severity'] == 'critical'
        
        # Test non-existent alert
        empty_history = self.escalation_service.get_escalation_history("non_existent")
        assert len(empty_history) == 0
    
    def test_format_supervisor_notification(self):
        """Test supervisor notification message formatting."""
        message = self.escalation_service._format_supervisor_notification(
            self.test_alert, 
            "supervisor_001"
        )
        
        assert "ESCALATED BORDER SECURITY ALERT" in message
        assert "SUPERVISOR APPROVAL REQUIRED" in message
        assert "alert_001" in message
        assert "Virtual Line Crossing" in message
        assert "CRITICAL" in message
        assert "0.9" in message  # risk score
        assert "cam_001" in message
        assert "supervisor_001" in message
    
    def test_escalation_rule_error_handling(self):
        """Test error handling in escalation rule processing."""
        def failing_condition(alert):
            raise Exception("Condition check failed")
        
        def failing_action(alert):
            raise Exception("Action failed")
        
        # Register rule with failing condition
        self.escalation_service.register_escalation_rule(
            "failing_rule",
            failing_condition,
            failing_action
        )
        
        # Should not raise exception, should handle gracefully
        self.escalation_service.process_escalation(self.test_alert)
        
        # No escalation history should be created for failed rule
        history = self.escalation_service.get_escalation_history(self.test_alert.id)
        failing_escalations = [h for h in history if h.get('rule') == 'failing_rule']
        assert len(failing_escalations) == 0