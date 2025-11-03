"""
Evidence management interfaces for Project Argus.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, BinaryIO
from datetime import datetime

from ..models import Evidence, ChainOfCustody, EvidenceType, EvidenceStatus


class IEvidenceStore(ABC):
    """Interface for immutable evidence storage."""
    
    @abstractmethod
    def store_evidence(self, file_data: bytes, evidence_type: EvidenceType, 
                      metadata: Dict[str, Any], created_by: str) -> str:
        """Store evidence and return evidence ID."""
        pass
    
    @abstractmethod
    def retrieve_evidence(self, evidence_id: str) -> Optional[Evidence]:
        """Retrieve evidence metadata by ID."""
        pass
    
    @abstractmethod
    def get_evidence_file(self, evidence_id: str) -> Optional[BinaryIO]:
        """Get evidence file content."""
        pass
    
    @abstractmethod
    def verify_integrity(self, evidence_id: str) -> bool:
        """Verify evidence integrity using stored hash."""
        pass
    
    @abstractmethod
    def seal_evidence(self, evidence_id: str, operator_id: str) -> bool:
        """Seal evidence to prevent modifications."""
        pass
    
    @abstractmethod
    def transfer_custody(self, evidence_id: str, from_operator: str, 
                        to_operator: str, reason: str) -> bool:
        """Transfer evidence custody between operators."""
        pass
    
    @abstractmethod
    def get_chain_of_custody(self, evidence_id: str) -> Optional[ChainOfCustody]:
        """Get complete chain of custody for evidence."""
        pass
    
    @abstractmethod
    def search_evidence(self, filters: Dict[str, Any]) -> List[Evidence]:
        """Search evidence by various criteria."""
        pass
    
    @abstractmethod
    def schedule_purge(self, evidence_id: str, purge_date: datetime, operator_id: str) -> bool:
        """Schedule evidence for automatic purging."""
        pass
    
    @abstractmethod
    def purge_expired_evidence(self) -> List[str]:
        """Purge expired evidence and return list of purged IDs."""
        pass


class IForensicsEngine(ABC):
    """Interface for automated forensics reporting."""
    
    @abstractmethod
    def generate_incident_report(self, incident_id: str, format: str = "pdf") -> str:
        """Generate comprehensive incident report."""
        pass
    
    @abstractmethod
    def create_video_summary(self, evidence_ids: List[str], 
                           start_time: datetime, end_time: datetime) -> str:
        """Create video summary from multiple evidence files."""
        pass
    
    @abstractmethod
    def extract_metadata(self, evidence_id: str) -> Dict[str, Any]:
        """Extract detailed metadata from evidence file."""
        pass
    
    @abstractmethod
    def create_timeline(self, incident_id: str) -> Dict[str, Any]:
        """Create chronological timeline of incident events."""
        pass
    
    @abstractmethod
    def export_legal_package(self, incident_id: str) -> str:
        """Export complete legal package with all evidence and documentation."""
        pass
    
    @abstractmethod
    def generate_chain_of_custody_report(self, evidence_id: str) -> str:
        """Generate detailed chain of custody report."""
        pass
    
    @abstractmethod
    def validate_evidence_integrity(self, evidence_ids: List[str]) -> Dict[str, bool]:
        """Validate integrity of multiple evidence files."""
        pass


class IAuditLogger(ABC):
    """Interface for comprehensive audit logging."""
    
    @abstractmethod
    def log_user_action(self, user_id: str, action: str, resource_type: str, 
                       resource_id: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log user action for audit trail."""
        pass
    
    @abstractmethod
    def log_system_event(self, event_type: str, component: str, 
                        details: Dict[str, Any]) -> None:
        """Log system event for audit trail."""
        pass
    
    @abstractmethod
    def log_security_event(self, event_type: str, user_id: Optional[str], 
                          source_ip: str, details: Dict[str, Any]) -> None:
        """Log security-related event."""
        pass
    
    @abstractmethod
    def log_evidence_access(self, evidence_id: str, user_id: str, 
                           action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log evidence access for chain of custody."""
        pass
    
    @abstractmethod
    def get_audit_logs(self, start_time: datetime, end_time: datetime, 
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve audit logs for time range with optional filters."""
        pass
    
    @abstractmethod
    def search_audit_logs(self, query: str, time_range_hours: int = 24) -> List[Dict[str, Any]]:
        """Search audit logs by text query."""
        pass
    
    @abstractmethod
    def export_audit_logs(self, start_time: datetime, end_time: datetime, 
                         format: str = "csv") -> str:
        """Export audit logs to file."""
        pass
    
    @abstractmethod
    def get_user_activity_summary(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get summary of user activity over time period."""
        pass


class IPrivacyManager(ABC):
    """Interface for privacy-preserving data management."""
    
    @abstractmethod
    def schedule_automatic_purge(self, incident_id: str) -> bool:
        """Schedule automatic purge for unconfirmed incidents."""
        pass
    
    @abstractmethod
    def cancel_automatic_purge(self, incident_id: str, operator_id: str) -> bool:
        """Cancel automatic purge when incident is confirmed."""
        pass
    
    @abstractmethod
    def anonymize_evidence(self, evidence_id: str, anonymization_level: str = "standard") -> str:
        """Create anonymized version of evidence."""
        pass
    
    @abstractmethod
    def apply_retention_policy(self, policy_name: str, filters: Dict[str, Any]) -> List[str]:
        """Apply retention policy to evidence matching filters."""
        pass
    
    @abstractmethod
    def generate_privacy_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate privacy compliance report for specified time period."""
        pass
    
    @abstractmethod
    def run_privacy_maintenance(self) -> Dict[str, Any]:
        """Run periodic privacy maintenance tasks."""
        pass