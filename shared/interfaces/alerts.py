"""
Alert management interfaces for Project Argus.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable

from ..models import Alert, CrossingEvent, Detection, Track


class IAlertEngine(ABC):
    """Interface for alert generation and management."""
    
    @abstractmethod
    def generate_alert(self, crossing_event: CrossingEvent, detection: Detection) -> Alert:
        """Generate an alert from a crossing event."""
        pass
    
    @abstractmethod
    def calculate_risk_score(self, event: CrossingEvent, context: Dict[str, Any]) -> float:
        """Calculate risk score for an event."""
        pass
    
    @abstractmethod
    def route_alert(self, alert: Alert) -> None:
        """Route alert to appropriate handlers/operators."""
        pass
    
    @abstractmethod
    def acknowledge_alert(self, alert_id: str, operator_id: str) -> bool:
        """Acknowledge an alert."""
        pass
    
    @abstractmethod
    def escalate_alert(self, alert_id: str, operator_id: str, reason: str) -> bool:
        """Escalate an alert to higher priority."""
        pass
    
    @abstractmethod
    def get_active_alerts(self, camera_id: Optional[str] = None) -> List[Alert]:
        """Get all active alerts, optionally filtered by camera."""
        pass


class IEscalationService(ABC):
    """Interface for alert escalation workflows."""
    
    @abstractmethod
    def register_escalation_rule(self, rule_name: str, condition: Callable[[Alert], bool], 
                               action: Callable[[Alert], None]) -> None:
        """Register an escalation rule."""
        pass
    
    @abstractmethod
    def process_escalation(self, alert: Alert) -> None:
        """Process escalation for an alert based on rules."""
        pass
    
    @abstractmethod
    def notify_supervisor(self, alert: Alert, supervisor_id: str) -> bool:
        """Notify supervisor about escalated alert."""
        pass
    
    @abstractmethod
    def get_escalation_history(self, alert_id: str) -> List[Dict[str, Any]]:
        """Get escalation history for an alert."""
        pass


class INotificationService(ABC):
    """Interface for multi-channel notification delivery."""
    
    @abstractmethod
    def send_email(self, recipient: str, subject: str, body: str, 
                  attachments: Optional[List[str]] = None) -> bool:
        """Send email notification."""
        pass
    
    @abstractmethod
    def send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS notification."""
        pass
    
    @abstractmethod
    def send_push_notification(self, user_id: str, title: str, message: str, 
                             data: Optional[Dict[str, Any]] = None) -> bool:
        """Send push notification to mobile app."""
        pass
    
    @abstractmethod
    def send_webhook(self, url: str, payload: Dict[str, Any]) -> bool:
        """Send webhook notification to external system."""
        pass
    
    @abstractmethod
    def notify_alert(self, alert: Alert, recipients: List[str], 
                    channels: List[str]) -> Dict[str, bool]:
        """Send alert notification through multiple channels."""
        pass
    
    @abstractmethod
    def get_notification_status(self, notification_id: str) -> Dict[str, Any]:
        """Get delivery status of a notification."""
        pass