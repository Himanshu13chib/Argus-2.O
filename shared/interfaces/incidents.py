"""
Incident management interfaces for Project Argus.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..models import Incident, Alert, Note, Resolution, IncidentStatus


class IIncidentManager(ABC):
    """Interface for incident case management."""
    
    @abstractmethod
    def create_incident(self, alert: Alert, operator_id: str, 
                       title: Optional[str] = None, description: Optional[str] = None) -> Incident:
        """Create a new incident from an alert."""
        pass
    
    @abstractmethod
    def update_incident(self, incident_id: str, updates: Dict[str, Any]) -> Optional[Incident]:
        """Update incident with new information."""
        pass
    
    @abstractmethod
    def add_note(self, incident_id: str, content: str, author_id: str, 
                note_type: str = "general") -> Optional[Note]:
        """Add a note to an incident."""
        pass
    
    @abstractmethod
    def add_evidence(self, incident_id: str, evidence_id: str) -> bool:
        """Link evidence to an incident."""
        pass
    
    @abstractmethod
    def assign_incident(self, incident_id: str, operator_id: str) -> bool:
        """Assign incident to an operator."""
        pass
    
    @abstractmethod
    def escalate_incident(self, incident_id: str, supervisor_id: str, reason: str) -> bool:
        """Escalate incident to supervisor."""
        pass
    
    @abstractmethod
    def resolve_incident(self, incident_id: str, resolution: Resolution) -> bool:
        """Resolve an incident with resolution details."""
        pass
    
    @abstractmethod
    def close_incident(self, incident_id: str, operator_id: str) -> bool:
        """Close a resolved incident."""
        pass
    
    @abstractmethod
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID."""
        pass
    
    @abstractmethod
    def get_incidents_by_status(self, status: IncidentStatus) -> List[Incident]:
        """Get all incidents with specific status."""
        pass
    
    @abstractmethod
    def get_incidents_by_operator(self, operator_id: str) -> List[Incident]:
        """Get all incidents assigned to an operator."""
        pass
    
    @abstractmethod
    def get_overdue_incidents(self, sla_hours: int = 24) -> List[Incident]:
        """Get incidents that are overdue based on SLA."""
        pass
    
    @abstractmethod
    def search_incidents(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Incident]:
        """Search incidents by text query and filters."""
        pass
    
    @abstractmethod
    def get_incident_statistics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get incident statistics for a time range."""
        pass