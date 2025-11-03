"""
Notification Service implementation for Project Argus.
Handles multi-channel alert delivery and communication.
"""

import logging
import smtplib
import json
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    from email.mime.base import MimeBase
    from email import encoders
except ImportError:
    # Fallback for systems where email modules are not available
    MimeText = None
    MimeMultipart = None
    MimeBase = None
    encoders = None
import requests
import uuid

from shared.interfaces.alerts import INotificationService, IEscalationService
from shared.models import Alert


logger = logging.getLogger(__name__)


class NotificationService(INotificationService):
    """
    Multi-channel notification delivery service.
    Implements requirements 7.5, 11.3.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.notification_history: Dict[str, Dict[str, Any]] = {}
        
        # Email configuration
        self.smtp_server = self.config.get('smtp_server', 'localhost')
        self.smtp_port = self.config.get('smtp_port', 587)
        self.smtp_username = self.config.get('smtp_username', '')
        self.smtp_password = self.config.get('smtp_password', '')
        self.from_email = self.config.get('from_email', 'alerts@projectargus.com')
        
        # SMS configuration (placeholder for integration)
        self.sms_api_url = self.config.get('sms_api_url', '')
        self.sms_api_key = self.config.get('sms_api_key', '')
        
        # Push notification configuration
        self.push_api_url = self.config.get('push_api_url', '')
        self.push_api_key = self.config.get('push_api_key', '')
        
        # Webhook configuration
        self.webhook_timeout = self.config.get('webhook_timeout', 30)
        
        logger.info("NotificationService initialized")
    
    def send_email(self, recipient: str, subject: str, body: str, 
                  attachments: Optional[List[str]] = None) -> bool:
        """
        Send email notification.
        Requirement 7.5: Implement supervisor approval workflows for critical responses.
        """
        try:
            if not MimeMultipart or not MimeText:
                logger.warning("Email modules not available, skipping email send")
                return False
                
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = recipient
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MimeText(body, 'html' if '<html>' in body.lower() else 'plain'))
            
            # Add attachments if provided
            if attachments and MimeBase and encoders:
                for file_path in attachments:
                    try:
                        with open(file_path, 'rb') as attachment:
                            part = MimeBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename= {file_path.split("/")[-1]}'
                            )
                            msg.attach(part)
                    except Exception as e:
                        logger.warning(f"Failed to attach file {file_path}: {e}")
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.smtp_username and self.smtp_password:
                    server.starttls()
                    server.login(self.smtp_username, self.smtp_password)
                
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {recipient}: {e}")
            return False
    
    def send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS notification."""
        try:
            if not self.sms_api_url or not self.sms_api_key:
                logger.warning("SMS configuration not available")
                return False
            
            payload = {
                'to': phone_number,
                'message': message,
                'api_key': self.sms_api_key
            }
            
            response = requests.post(
                self.sms_api_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"SMS sent successfully to {phone_number}")
                return True
            else:
                logger.error(f"SMS API returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send SMS to {phone_number}: {e}")
            return False
    
    def send_push_notification(self, user_id: str, title: str, message: str, 
                             data: Optional[Dict[str, Any]] = None) -> bool:
        """Send push notification to mobile app."""
        try:
            if not self.push_api_url or not self.push_api_key:
                logger.warning("Push notification configuration not available")
                return False
            
            payload = {
                'user_id': user_id,
                'title': title,
                'message': message,
                'data': data or {},
                'api_key': self.push_api_key
            }
            
            response = requests.post(
                self.push_api_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Push notification sent successfully to user {user_id}")
                return True
            else:
                logger.error(f"Push API returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send push notification to user {user_id}: {e}")
            return False
    
    def send_webhook(self, url: str, payload: Dict[str, Any]) -> bool:
        """Send webhook notification to external system."""
        try:
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'ProjectArgus/1.0'
            }
            
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.webhook_timeout
            )
            
            if 200 <= response.status_code < 300:
                logger.info(f"Webhook sent successfully to {url}")
                return True
            else:
                logger.error(f"Webhook to {url} returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send webhook to {url}: {e}")
            return False
    
    def notify_alert(self, alert: Alert, recipients: List[str], 
                    channels: List[str]) -> Dict[str, bool]:
        """
        Send alert notification through multiple channels.
        Requirement 11.3: Implement escalation procedures for high-risk incidents.
        """
        notification_id = str(uuid.uuid4())
        results = {}
        
        try:
            # Prepare alert content
            subject = f"Project Argus Alert - {alert.type.value.replace('_', ' ').title()}"
            message = self._format_alert_message(alert)
            html_message = self._format_alert_html(alert)
            
            # Send through each channel
            for channel in channels:
                channel_results = []
                
                if channel == 'email':
                    for recipient in recipients:
                        success = self.send_email(recipient, subject, html_message)
                        channel_results.append(success)
                
                elif channel == 'sms':
                    # SMS message should be shorter
                    sms_message = self._format_alert_sms(alert)
                    for recipient in recipients:
                        success = self.send_sms(recipient, sms_message)
                        channel_results.append(success)
                
                elif channel == 'push':
                    push_data = {
                        'alert_id': alert.id,
                        'camera_id': alert.camera_id,
                        'severity': alert.severity.value,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for recipient in recipients:
                        success = self.send_push_notification(recipient, subject, message, push_data)
                        channel_results.append(success)
                
                elif channel == 'webhook':
                    webhook_payload = {
                        'alert': alert.to_dict(),
                        'notification_id': notification_id,
                        'timestamp': datetime.now().isoformat()
                    }
                    for recipient in recipients:  # recipient is webhook URL in this case
                        success = self.send_webhook(recipient, webhook_payload)
                        channel_results.append(success)
                
                results[channel] = all(channel_results) if channel_results else False
            
            # Store notification history
            self.notification_history[notification_id] = {
                'alert_id': alert.id,
                'recipients': recipients,
                'channels': channels,
                'results': results,
                'timestamp': datetime.now(),
                'status': 'completed'
            }
            
            logger.info(f"Alert notification {notification_id} completed for alert {alert.id}")
            return results
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
            self.notification_history[notification_id] = {
                'alert_id': alert.id,
                'recipients': recipients,
                'channels': channels,
                'results': results,
                'timestamp': datetime.now(),
                'status': 'failed',
                'error': str(e)
            }
            return results
    
    def get_notification_status(self, notification_id: str) -> Dict[str, Any]:
        """Get delivery status of a notification."""
        return self.notification_history.get(notification_id, {
            'status': 'not_found',
            'message': 'Notification ID not found'
        })
    
    def _format_alert_message(self, alert: Alert) -> str:
        """Format alert for text-based notifications."""
        message = f"""
BORDER SECURITY ALERT

Type: {alert.type.value.replace('_', ' ').title()}
Severity: {alert.severity.value.upper()}
Camera: {alert.camera_id}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Confidence: {alert.confidence:.2f}
Risk Score: {alert.risk_score:.2f}
"""
        
        if alert.gps_coordinates:
            message += f"Location: {alert.gps_coordinates[0]:.6f}, {alert.gps_coordinates[1]:.6f}\n"
        
        if alert.crossing_event:
            message += f"Crossing Direction: {alert.crossing_event.crossing_direction}\n"
        
        message += f"\nAlert ID: {alert.id}"
        
        return message.strip()
    
    def _format_alert_html(self, alert: Alert) -> str:
        """Format alert for HTML email notifications."""
        severity_colors = {
            'low': '#28a745',
            'medium': '#ffc107', 
            'high': '#fd7e14',
            'critical': '#dc3545'
        }
        
        color = severity_colors.get(alert.severity.value, '#6c757d')
        
        html = f"""
<html>
<body style="font-family: Arial, sans-serif; margin: 20px;">
    <div style="border-left: 4px solid {color}; padding-left: 20px;">
        <h2 style="color: {color}; margin-top: 0;">Border Security Alert</h2>
        
        <table style="border-collapse: collapse; width: 100%; max-width: 600px;">
            <tr>
                <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #ddd;">Type:</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.type.value.replace('_', ' ').title()}</td>
            </tr>
            <tr>
                <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #ddd;">Severity:</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd; color: {color}; font-weight: bold;">{alert.severity.value.upper()}</td>
            </tr>
            <tr>
                <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #ddd;">Camera:</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.camera_id}</td>
            </tr>
            <tr>
                <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #ddd;">Time:</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
            </tr>
            <tr>
                <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #ddd;">Confidence:</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.confidence:.2f}</td>
            </tr>
            <tr>
                <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #ddd;">Risk Score:</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.risk_score:.2f}</td>
            </tr>
"""
        
        if alert.gps_coordinates:
            html += f"""
            <tr>
                <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #ddd;">Location:</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.gps_coordinates[0]:.6f}, {alert.gps_coordinates[1]:.6f}</td>
            </tr>
"""
        
        if alert.crossing_event:
            html += f"""
            <tr>
                <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #ddd;">Crossing Direction:</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.crossing_event.crossing_direction}</td>
            </tr>
"""
        
        html += f"""
        </table>
        
        <p style="margin-top: 20px; font-size: 12px; color: #666;">
            Alert ID: {alert.id}<br>
            Generated by Project Argus Border Detection System
        </p>
    </div>
</body>
</html>
"""
        return html
    
    def _format_alert_sms(self, alert: Alert) -> str:
        """Format alert for SMS notifications (shorter format)."""
        return f"BORDER ALERT: {alert.type.value.replace('_', ' ').title()} at {alert.camera_id} - {alert.severity.value.upper()} severity. Time: {alert.timestamp.strftime('%H:%M')}. ID: {alert.id[:8]}"


class EscalationService(IEscalationService):
    """
    Alert escalation workflow service.
    Implements requirements 7.5, 11.3.
    """
    
    def __init__(self, notification_service: NotificationService):
        self.notification_service = notification_service
        self.escalation_rules: Dict[str, Dict[str, Any]] = {}
        self.escalation_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Default escalation rules
        self._setup_default_rules()
        
        logger.info("EscalationService initialized")
    
    def register_escalation_rule(self, rule_name: str, condition: callable, action: callable) -> None:
        """Register an escalation rule."""
        self.escalation_rules[rule_name] = {
            'condition': condition,
            'action': action,
            'created_at': datetime.now()
        }
        logger.info(f"Registered escalation rule: {rule_name}")
    
    def process_escalation(self, alert: Alert) -> None:
        """
        Process escalation for an alert based on rules.
        Requirement 11.3: Implement escalation procedures for high-risk incidents.
        """
        try:
            escalations_triggered = []
            
            for rule_name, rule in self.escalation_rules.items():
                try:
                    if rule['condition'](alert):
                        rule['action'](alert)
                        escalations_triggered.append(rule_name)
                        
                        # Record escalation
                        if alert.id not in self.escalation_history:
                            self.escalation_history[alert.id] = []
                        
                        self.escalation_history[alert.id].append({
                            'rule': rule_name,
                            'timestamp': datetime.now(),
                            'alert_severity': alert.severity.value,
                            'alert_risk_score': alert.risk_score
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing escalation rule {rule_name}: {e}")
            
            if escalations_triggered:
                logger.info(f"Escalation rules triggered for alert {alert.id}: {escalations_triggered}")
            
        except Exception as e:
            logger.error(f"Error processing escalation for alert {alert.id}: {e}")
    
    def notify_supervisor(self, alert: Alert, supervisor_id: str) -> bool:
        """
        Notify supervisor about escalated alert.
        Requirement 7.5: Implement supervisor approval workflows for critical responses.
        """
        try:
            # Get supervisor contact information (placeholder - would integrate with user management)
            supervisor_contacts = self._get_supervisor_contacts(supervisor_id)
            
            if not supervisor_contacts:
                logger.warning(f"No contact information found for supervisor {supervisor_id}")
                return False
            
            # Prepare escalation message
            subject = f"ESCALATED ALERT - Supervisor Approval Required"
            message = self._format_supervisor_notification(alert, supervisor_id)
            
            # Send notifications through multiple channels for critical alerts
            channels = ['email']
            if alert.severity == 'critical':
                channels.extend(['sms', 'push'])
            
            results = self.notification_service.notify_alert(
                alert, 
                supervisor_contacts.get('email', []),
                channels
            )
            
            success = any(results.values())
            
            if success:
                logger.info(f"Supervisor {supervisor_id} notified about escalated alert {alert.id}")
            else:
                logger.error(f"Failed to notify supervisor {supervisor_id} about alert {alert.id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error notifying supervisor {supervisor_id}: {e}")
            return False
    
    def get_escalation_history(self, alert_id: str) -> List[Dict[str, Any]]:
        """Get escalation history for an alert."""
        return self.escalation_history.get(alert_id, [])
    
    def _setup_default_rules(self) -> None:
        """Setup default escalation rules."""
        
        # Critical severity auto-escalation
        def critical_condition(alert: Alert) -> bool:
            return alert.severity.value == 'critical'
        
        def critical_action(alert: Alert) -> None:
            # Notify all supervisors for critical alerts
            supervisors = self._get_all_supervisors()
            for supervisor_id in supervisors:
                self.notify_supervisor(alert, supervisor_id)
        
        self.register_escalation_rule("critical_auto_escalation", critical_condition, critical_action)
        
        # High risk score escalation
        def high_risk_condition(alert: Alert) -> bool:
            return alert.risk_score >= 0.8
        
        def high_risk_action(alert: Alert) -> None:
            # Notify duty supervisor
            duty_supervisor = self._get_duty_supervisor()
            if duty_supervisor:
                self.notify_supervisor(alert, duty_supervisor)
        
        self.register_escalation_rule("high_risk_escalation", high_risk_condition, high_risk_action)
        
        # Multiple crossings escalation
        def multiple_crossings_condition(alert: Alert) -> bool:
            return alert.metadata.get('simultaneous_crossings', 0) > 2
        
        def multiple_crossings_action(alert: Alert) -> None:
            # Notify security team
            self._notify_security_team(alert)
        
        self.register_escalation_rule("multiple_crossings", multiple_crossings_condition, multiple_crossings_action)
    
    def _get_supervisor_contacts(self, supervisor_id: str) -> Dict[str, List[str]]:
        """Get supervisor contact information (placeholder implementation)."""
        # This would integrate with user management system
        return {
            'email': [f"{supervisor_id}@border-security.gov.in"],
            'sms': ['+91-XXXXXXXXXX'],
            'push': [supervisor_id]
        }
    
    def _get_all_supervisors(self) -> List[str]:
        """Get all supervisor IDs (placeholder implementation)."""
        return ['supervisor1', 'supervisor2', 'duty_officer']
    
    def _get_duty_supervisor(self) -> Optional[str]:
        """Get current duty supervisor (placeholder implementation)."""
        # This would check duty roster
        return 'duty_officer'
    
    def _notify_security_team(self, alert: Alert) -> None:
        """Notify security team about alert."""
        # Placeholder for security team notification
        logger.info(f"Security team notified about alert {alert.id}")
    
    def _format_supervisor_notification(self, alert: Alert, supervisor_id: str) -> str:
        """Format supervisor notification message."""
        return f"""
ESCALATED BORDER SECURITY ALERT - SUPERVISOR APPROVAL REQUIRED

Alert ID: {alert.id}
Type: {alert.type.value.replace('_', ' ').title()}
Severity: {alert.severity.value.upper()}
Risk Score: {alert.risk_score:.2f}
Camera: {alert.camera_id}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

This alert has been escalated and requires supervisor review and approval for any response actions.

Please review the alert details in the command center dashboard and authorize appropriate response measures.

Supervisor: {supervisor_id}
Escalation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""