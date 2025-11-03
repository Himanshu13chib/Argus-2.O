"""
Alert Engine implementation for Project Argus.
Handles real-time alert creation, risk scoring, and routing.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Callable
import asyncio
from collections import defaultdict

from shared.interfaces.alerts import IAlertEngine
from shared.models import Alert, CrossingEvent, Detection, Track
from shared.models.alerts import AlertType, Severity


logger = logging.getLogger(__name__)


class AlertEngine(IAlertEngine):
    """
    Real-time alert generation engine with risk scoring and routing capabilities.
    Implements requirements 1.3, 4.3, 7.4, 8.2.
    """
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.risk_factors: Dict[str, float] = {
            'night_hours': 0.3,
            'multiple_crossings': 0.2,
            'high_confidence': 0.2,
            'restricted_zone': 0.4,
            'repeat_offender': 0.3,
            'group_crossing': 0.25
        }
        
        # Track crossing patterns for risk assessment
        self.crossing_history: Dict[str, List[datetime]] = defaultdict(list)
        self.camera_activity: Dict[str, int] = defaultdict(int)
        
        logger.info("AlertEngine initialized")
    
    def generate_alert(self, crossing_event: CrossingEvent, detection: Detection) -> Alert:
        """
        Generate an alert from a crossing event.
        Requirement 1.3: Generate immediate alert with confidence score when person crosses virtual line.
        """
        try:
            # Create base alert
            alert = Alert(
                type=AlertType.VIRTUAL_LINE_CROSSING,
                camera_id=detection.camera_id,
                detection_id=detection.id,
                timestamp=crossing_event.timestamp,
                confidence=crossing_event.confidence,
                crossing_event=crossing_event,
                gps_coordinates=detection.metadata.get('gps_coordinates') if detection.metadata else None
            )
            
            # Calculate risk score with context
            context = self._build_context(crossing_event, detection)
            alert.calculate_risk_score(context)
            
            # Determine severity based on risk score
            alert.severity = self._determine_severity(alert.risk_score, context)
            
            # Store alert
            self.active_alerts[alert.id] = alert
            
            # Update tracking data
            self._update_crossing_history(detection.camera_id, crossing_event.timestamp)
            
            logger.info(f"Generated alert {alert.id} with risk score {alert.risk_score:.2f}")
            
            return alert
            
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
            raise
    
    def calculate_risk_score(self, event: CrossingEvent, context: Dict[str, Any]) -> float:
        """
        Calculate risk score based on crossing patterns and context.
        Requirement 8.2: Generate predictive risk scores when unusual patterns detected.
        """
        try:
            base_score = event.confidence * 0.4
            
            # Time-based risk assessment
            hour = event.timestamp.hour
            if 22 <= hour or hour <= 6:  # Night hours (10 PM - 6 AM)
                base_score += self.risk_factors['night_hours']
                logger.debug(f"Night hours detected, adding {self.risk_factors['night_hours']} to risk")
            
            # Multiple simultaneous crossings
            if context.get('simultaneous_crossings', 0) > 1:
                multiplier = min(3, context['simultaneous_crossings'])
                base_score += self.risk_factors['multiple_crossings'] * multiplier
                logger.debug(f"Multiple crossings detected: {context['simultaneous_crossings']}")
            
            # High confidence detection
            if event.confidence > 0.9:
                base_score += self.risk_factors['high_confidence']
            
            # Historical pattern analysis
            camera_id = context.get('camera_id', '')
            recent_crossings = self._get_recent_crossings(camera_id, hours=24)
            if len(recent_crossings) > 5:  # Unusual activity
                base_score += self.risk_factors['repeat_offender']
                logger.debug(f"High activity detected: {len(recent_crossings)} crossings in 24h")
            
            # Group crossing detection
            if context.get('group_size', 1) > 1:
                group_multiplier = min(2, context['group_size'] * 0.5)
                base_score += self.risk_factors['group_crossing'] * group_multiplier
            
            # Restricted zone crossing
            if context.get('restricted_zone', False):
                base_score += self.risk_factors['restricted_zone']
                logger.debug("Restricted zone crossing detected")
            
            # Environmental factors
            weather = context.get('weather_condition', 'clear')
            if weather in ['fog', 'heavy_rain', 'storm']:
                base_score += 0.1  # Slightly higher risk in poor visibility
            
            # Cap the risk score at 1.0
            final_score = min(1.0, base_score)
            
            logger.debug(f"Calculated risk score: {final_score:.3f} for event {event.detection_id}")
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return event.confidence * 0.5  # Fallback score
    
    def route_alert(self, alert: Alert) -> None:
        """
        Route alert to appropriate handlers based on severity and type.
        Requirement 4.3: Display alert details with recommended actions.
        """
        try:
            logger.info(f"Routing alert {alert.id} (severity: {alert.severity.value})")
            
            # Route to all registered handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")
            
            # Special routing for high-severity alerts
            if alert.severity in [Severity.HIGH, Severity.CRITICAL]:
                self._route_high_priority_alert(alert)
            
            logger.info(f"Alert {alert.id} routed to {len(self.alert_handlers)} handlers")
            
        except Exception as e:
            logger.error(f"Error routing alert {alert.id}: {e}")
    
    def acknowledge_alert(self, alert_id: str, operator_id: str) -> bool:
        """Acknowledge an alert."""
        try:
            if alert_id not in self.active_alerts:
                logger.warning(f"Alert {alert_id} not found for acknowledgment")
                return False
            
            alert = self.active_alerts[alert_id]
            alert.acknowledge(operator_id)
            
            logger.info(f"Alert {alert_id} acknowledged by operator {operator_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    def escalate_alert(self, alert_id: str, operator_id: str, reason: str) -> bool:
        """
        Escalate an alert to higher priority.
        Requirement 7.4: Provide escalation procedures for high-risk incidents.
        """
        try:
            if alert_id not in self.active_alerts:
                logger.warning(f"Alert {alert_id} not found for escalation")
                return False
            
            alert = self.active_alerts[alert_id]
            original_severity = alert.severity
            alert.escalate(operator_id, reason)
            
            # Re-route escalated alert
            self.route_alert(alert)
            
            logger.info(f"Alert {alert_id} escalated from {original_severity.value} to {alert.severity.value} by {operator_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error escalating alert {alert_id}: {e}")
            return False
    
    def get_active_alerts(self, camera_id: Optional[str] = None) -> List[Alert]:
        """Get all active alerts, optionally filtered by camera."""
        try:
            alerts = list(self.active_alerts.values())
            
            if camera_id:
                alerts = [alert for alert in alerts if alert.camera_id == camera_id]
            
            # Sort by timestamp (newest first) and severity
            severity_order = {Severity.CRITICAL: 4, Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1}
            alerts.sort(key=lambda a: (severity_order[a.severity], a.timestamp), reverse=True)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    def register_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register a handler for alert routing."""
        self.alert_handlers.append(handler)
        logger.info(f"Registered alert handler: {handler.__name__}")
    
    def remove_alert(self, alert_id: str) -> bool:
        """Remove an alert from active alerts."""
        try:
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
                logger.info(f"Removed alert {alert_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing alert {alert_id}: {e}")
            return False
    
    def _build_context(self, crossing_event: CrossingEvent, detection: Detection) -> Dict[str, Any]:
        """Build context data for risk assessment."""
        context = {
            'camera_id': detection.camera_id,
            'timestamp': crossing_event.timestamp,
            'confidence': crossing_event.confidence
        }
        
        # Check for simultaneous crossings
        recent_alerts = [
            alert for alert in self.active_alerts.values()
            if (crossing_event.timestamp - alert.timestamp).total_seconds() < 30
            and alert.camera_id == detection.camera_id
        ]
        context['simultaneous_crossings'] = len(recent_alerts)
        
        # Add historical crossing data
        recent_crossings = self._get_recent_crossings(detection.camera_id, hours=24)
        context['historical_crossings'] = len(recent_crossings)
        
        return context
    
    def _determine_severity(self, risk_score: float, context: Dict[str, Any]) -> Severity:
        """Determine alert severity based on risk score and context."""
        if risk_score >= 0.8:
            return Severity.CRITICAL
        elif risk_score >= 0.6:
            return Severity.HIGH
        elif risk_score >= 0.4:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def _route_high_priority_alert(self, alert: Alert) -> None:
        """Special routing for high-priority alerts."""
        logger.warning(f"HIGH PRIORITY ALERT: {alert.id} - {alert.type.value}")
        
        # Add to metadata for special handling
        alert.metadata['requires_immediate_attention'] = True
        alert.metadata['auto_escalated'] = True
    
    def _update_crossing_history(self, camera_id: str, timestamp: datetime) -> None:
        """Update crossing history for pattern analysis."""
        self.crossing_history[camera_id].append(timestamp)
        
        # Keep only last 7 days of history
        cutoff = timestamp - timedelta(days=7)
        self.crossing_history[camera_id] = [
            ts for ts in self.crossing_history[camera_id] if ts > cutoff
        ]
    
    def _get_recent_crossings(self, camera_id: str, hours: int = 24) -> List[datetime]:
        """Get recent crossings for a camera within specified hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            ts for ts in self.crossing_history[camera_id] if ts > cutoff
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert engine statistics."""
        total_alerts = len(self.active_alerts)
        severity_counts = defaultdict(int)
        
        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1
        
        return {
            'total_active_alerts': total_alerts,
            'severity_breakdown': dict(severity_counts),
            'total_cameras_with_activity': len(self.crossing_history),
            'alert_handlers_registered': len(self.alert_handlers)
        }