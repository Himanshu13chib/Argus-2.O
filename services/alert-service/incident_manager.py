"""
Incident Manager implementation for Project Argus.
Handles case lifecycle management, operator assignment, and workflow automation.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from collections import defaultdict
import re

from shared.interfaces.incidents import IIncidentManager
from shared.models import Incident, Alert, Note, Resolution, IncidentStatus
from shared.models.incidents import IncidentPriority, ResolutionType


logger = logging.getLogger(__name__)


class IncidentManager(IIncidentManager):
    """
    Incident management system for case lifecycle management.
    Implements requirements 4.5, 7.1, 7.2, 7.4.
    """
    
    def __init__(self):
        self.incidents: Dict[str, Incident] = {}
        self.operator_assignments: Dict[str, List[str]] = defaultdict(list)  # operator_id -> incident_ids
        self.auto_assignment_rules: List[Dict[str, Any]] = []
        
        # SLA configuration
        self.sla_hours = {
            IncidentPriority.CRITICAL: 2,
            IncidentPriority.HIGH: 8,
            IncidentPriority.MEDIUM: 24,
            IncidentPriority.LOW: 72
        }
        
        logger.info("IncidentManager initialized")
    
    def create_incident(self, alert: Alert, operator_id: str, 
                       title: Optional[str] = None, description: Optional[str] = None) -> Incident:
        """
        Create a new incident from an alert.
        Requirement 4.5: Enable operators to create and annotate incident cases.
        """
        try:
            # Generate title if not provided
            if not title:
                title = f"Border Crossing Alert - {alert.type.value.replace('_', ' ').title()}"
            
            # Generate description if not provided
            if not description:
                description = f"Incident created from alert {alert.id} at camera {alert.camera_id}"
                if alert.crossing_event:
                    description += f" - {alert.crossing_event.crossing_direction} crossing detected"
            
            # Determine priority based on alert severity and risk score
            priority = self._determine_priority(alert)
            
            # Create incident
            incident = Incident(
                alert_id=alert.id,
                operator_id=operator_id,
                title=title,
                description=description,
                priority=priority,
                location=f"Camera {alert.camera_id}",
                gps_coordinates=alert.gps_coordinates,
                camera_ids=[alert.camera_id]
            )
            
            # Add initial metadata from alert
            incident.metadata.update({
                'alert_confidence': alert.confidence,
                'alert_risk_score': alert.risk_score,
                'alert_type': alert.type.value,
                'creation_source': 'alert_conversion'
            })
            
            # Add initial note
            incident.add_note(
                content=f"Incident created from alert {alert.id} by operator {operator_id}",
                author_id=operator_id,
                note_type="action"
            )
            
            # Store incident
            self.incidents[incident.id] = incident
            
            # Auto-assign if rules exist
            self._apply_auto_assignment(incident)
            
            logger.info(f"Created incident {incident.id} from alert {alert.id}")
            
            return incident
            
        except Exception as e:
            logger.error(f"Error creating incident from alert {alert.id}: {e}")
            raise
    
    def update_incident(self, incident_id: str, updates: Dict[str, Any]) -> Optional[Incident]:
        """
        Update incident with new information.
        Requirement 7.2: Implement incident annotation and resolution tracking.
        """
        try:
            if incident_id not in self.incidents:
                logger.warning(f"Incident {incident_id} not found for update")
                return None
            
            incident = self.incidents[incident_id]
            original_status = incident.status
            
            # Apply updates
            for field, value in updates.items():
                if hasattr(incident, field):
                    setattr(incident, field, value)
                else:
                    incident.metadata[field] = value
            
            incident.updated_at = datetime.now()
            
            # Log status changes
            if 'status' in updates and updates['status'] != original_status:
                incident.add_note(
                    content=f"Status changed from {original_status.value} to {updates['status'].value}",
                    author_id=updates.get('updated_by', 'system'),
                    note_type="action"
                )
            
            logger.info(f"Updated incident {incident_id}")
            return incident
            
        except Exception as e:
            logger.error(f"Error updating incident {incident_id}: {e}")
            return None
    
    def add_note(self, incident_id: str, content: str, author_id: str, 
                note_type: str = "general") -> Optional[Note]:
        """Add a note to an incident."""
        try:
            if incident_id not in self.incidents:
                logger.warning(f"Incident {incident_id} not found for note addition")
                return None
            
            incident = self.incidents[incident_id]
            note = incident.add_note(content, author_id, note_type)
            
            logger.info(f"Added note to incident {incident_id} by {author_id}")
            return note
            
        except Exception as e:
            logger.error(f"Error adding note to incident {incident_id}: {e}")
            return None
    
    def add_evidence(self, incident_id: str, evidence_id: str) -> bool:
        """Link evidence to an incident."""
        try:
            if incident_id not in self.incidents:
                logger.warning(f"Incident {incident_id} not found for evidence addition")
                return False
            
            incident = self.incidents[incident_id]
            incident.add_evidence(evidence_id)
            
            # Add note about evidence
            incident.add_note(
                content=f"Evidence {evidence_id} linked to incident",
                author_id="system",
                note_type="evidence"
            )
            
            logger.info(f"Added evidence {evidence_id} to incident {incident_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding evidence to incident {incident_id}: {e}")
            return False
    
    def assign_incident(self, incident_id: str, operator_id: str) -> bool:
        """
        Assign incident to an operator.
        Requirement 7.1: Implement operator assignment and workflow automation.
        """
        try:
            if incident_id not in self.incidents:
                logger.warning(f"Incident {incident_id} not found for assignment")
                return False
            
            incident = self.incidents[incident_id]
            old_assignee = incident.assigned_to
            
            incident.assign_to(operator_id)
            
            # Update operator assignments tracking
            if old_assignee:
                self.operator_assignments[old_assignee].remove(incident_id)
            self.operator_assignments[operator_id].append(incident_id)
            
            # Add assignment note
            incident.add_note(
                content=f"Incident assigned to operator {operator_id}",
                author_id="system",
                note_type="action"
            )
            
            logger.info(f"Assigned incident {incident_id} to operator {operator_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error assigning incident {incident_id}: {e}")
            return False
    
    def escalate_incident(self, incident_id: str, supervisor_id: str, reason: str) -> bool:
        """
        Escalate incident to supervisor.
        Requirement 7.4: Provide escalation procedures for high-risk incidents.
        """
        try:
            if incident_id not in self.incidents:
                logger.warning(f"Incident {incident_id} not found for escalation")
                return False
            
            incident = self.incidents[incident_id]
            incident.escalate(supervisor_id, reason)
            
            # Add escalation note
            incident.add_note(
                content=f"Incident escalated to supervisor {supervisor_id}. Reason: {reason}",
                author_id="system",
                note_type="escalation"
            )
            
            logger.warning(f"Escalated incident {incident_id} to supervisor {supervisor_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error escalating incident {incident_id}: {e}")
            return False
    
    def resolve_incident(self, incident_id: str, resolution: Resolution) -> bool:
        """Resolve an incident with resolution details."""
        try:
            if incident_id not in self.incidents:
                logger.warning(f"Incident {incident_id} not found for resolution")
                return False
            
            incident = self.incidents[incident_id]
            incident.resolve(resolution)
            
            # Add resolution note
            incident.add_note(
                content=f"Incident resolved: {resolution.type.value} - {resolution.description}",
                author_id=resolution.resolved_by,
                note_type="resolution"
            )
            
            logger.info(f"Resolved incident {incident_id} as {resolution.type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving incident {incident_id}: {e}")
            return False
    
    def close_incident(self, incident_id: str, operator_id: str) -> bool:
        """Close a resolved incident."""
        try:
            if incident_id not in self.incidents:
                logger.warning(f"Incident {incident_id} not found for closure")
                return False
            
            incident = self.incidents[incident_id]
            
            # Ensure incident is resolved before closing
            if incident.status != IncidentStatus.RESOLVED:
                logger.warning(f"Cannot close incident {incident_id} - not resolved")
                return False
            
            incident.close(operator_id)
            
            # Remove from operator assignments
            if incident.assigned_to and incident.assigned_to in self.operator_assignments:
                if incident_id in self.operator_assignments[incident.assigned_to]:
                    self.operator_assignments[incident.assigned_to].remove(incident_id)
            
            logger.info(f"Closed incident {incident_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing incident {incident_id}: {e}")
            return False
    
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID."""
        return self.incidents.get(incident_id)
    
    def get_incidents_by_status(self, status: IncidentStatus) -> List[Incident]:
        """Get all incidents with specific status."""
        return [incident for incident in self.incidents.values() if incident.status == status]
    
    def get_incidents_by_operator(self, operator_id: str) -> List[Incident]:
        """Get all incidents assigned to an operator."""
        return [
            self.incidents[incident_id] 
            for incident_id in self.operator_assignments.get(operator_id, [])
            if incident_id in self.incidents
        ]
    
    def get_overdue_incidents(self, sla_hours: int = 24) -> List[Incident]:
        """Get incidents that are overdue based on SLA."""
        overdue = []
        for incident in self.incidents.values():
            # Use priority-specific SLA if available
            priority_sla = self.sla_hours.get(incident.priority, sla_hours)
            if incident.is_overdue(priority_sla):
                overdue.append(incident)
        
        return sorted(overdue, key=lambda i: i.created_at)
    
    def search_incidents(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Incident]:
        """Search incidents by text query and filters."""
        try:
            results = []
            query_lower = query.lower()
            
            for incident in self.incidents.values():
                # Text search in title, description, and notes
                if (query_lower in incident.title.lower() or 
                    query_lower in incident.description.lower() or
                    any(query_lower in note.content.lower() for note in incident.notes)):
                    
                    # Apply filters if provided
                    if filters:
                        if not self._matches_filters(incident, filters):
                            continue
                    
                    results.append(incident)
            
            # Sort by relevance (most recent first)
            results.sort(key=lambda i: i.updated_at, reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Error searching incidents: {e}")
            return []
    
    def get_incident_statistics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get incident statistics for a time range."""
        try:
            cutoff = datetime.now() - timedelta(hours=time_range_hours)
            recent_incidents = [
                incident for incident in self.incidents.values()
                if incident.created_at > cutoff
            ]
            
            # Status breakdown
            status_counts = defaultdict(int)
            priority_counts = defaultdict(int)
            resolution_counts = defaultdict(int)
            
            total_incidents = len(recent_incidents)
            resolved_incidents = 0
            avg_resolution_time = 0
            
            for incident in recent_incidents:
                status_counts[incident.status.value] += 1
                priority_counts[incident.priority.value] += 1
                
                if incident.resolution:
                    resolution_counts[incident.resolution.type.value] += 1
                    resolved_incidents += 1
                    
                    if incident.closed_at:
                        duration = (incident.closed_at - incident.created_at).total_seconds() / 3600
                        avg_resolution_time += duration
            
            if resolved_incidents > 0:
                avg_resolution_time /= resolved_incidents
            
            # Overdue incidents
            overdue_incidents = len(self.get_overdue_incidents())
            
            return {
                'time_range_hours': time_range_hours,
                'total_incidents': total_incidents,
                'resolved_incidents': resolved_incidents,
                'overdue_incidents': overdue_incidents,
                'resolution_rate': resolved_incidents / total_incidents if total_incidents > 0 else 0,
                'avg_resolution_time_hours': round(avg_resolution_time, 2),
                'status_breakdown': dict(status_counts),
                'priority_breakdown': dict(priority_counts),
                'resolution_breakdown': dict(resolution_counts),
                'active_operators': len([op for op, incidents in self.operator_assignments.items() if incidents])
            }
            
        except Exception as e:
            logger.error(f"Error getting incident statistics: {e}")
            return {}
    
    def add_auto_assignment_rule(self, rule: Dict[str, Any]) -> None:
        """Add an automatic assignment rule."""
        self.auto_assignment_rules.append(rule)
        logger.info(f"Added auto-assignment rule: {rule.get('name', 'unnamed')}")
    
    def _determine_priority(self, alert: Alert) -> IncidentPriority:
        """Determine incident priority based on alert properties."""
        if alert.risk_score >= 0.8:
            return IncidentPriority.CRITICAL
        elif alert.risk_score >= 0.6:
            return IncidentPriority.HIGH
        elif alert.risk_score >= 0.4:
            return IncidentPriority.MEDIUM
        else:
            return IncidentPriority.LOW
    
    def _apply_auto_assignment(self, incident: Incident) -> None:
        """Apply automatic assignment rules to an incident."""
        try:
            for rule in self.auto_assignment_rules:
                if self._matches_assignment_rule(incident, rule):
                    operator_id = rule.get('operator_id')
                    if operator_id:
                        self.assign_incident(incident.id, operator_id)
                        logger.info(f"Auto-assigned incident {incident.id} to {operator_id} via rule {rule.get('name')}")
                        break
        except Exception as e:
            logger.error(f"Error applying auto-assignment rules: {e}")
    
    def _matches_assignment_rule(self, incident: Incident, rule: Dict[str, Any]) -> bool:
        """Check if incident matches an assignment rule."""
        conditions = rule.get('conditions', {})
        
        # Check priority condition
        if 'priority' in conditions:
            if incident.priority.value not in conditions['priority']:
                return False
        
        # Check camera condition
        if 'camera_ids' in conditions:
            if not any(cam_id in conditions['camera_ids'] for cam_id in incident.camera_ids):
                return False
        
        # Check time condition
        if 'time_range' in conditions:
            hour = incident.created_at.hour
            time_range = conditions['time_range']
            if not (time_range[0] <= hour <= time_range[1]):
                return False
        
        return True
    
    def _matches_filters(self, incident: Incident, filters: Dict[str, Any]) -> bool:
        """Check if incident matches search filters."""
        for key, value in filters.items():
            if key == 'status' and incident.status.value != value:
                return False
            elif key == 'priority' and incident.priority.value != value:
                return False
            elif key == 'operator_id' and incident.assigned_to != value:
                return False
            elif key == 'camera_id' and value not in incident.camera_ids:
                return False
            elif key == 'date_from':
                if incident.created_at < datetime.fromisoformat(value):
                    return False
            elif key == 'date_to':
                if incident.created_at > datetime.fromisoformat(value):
                    return False
        
        return True