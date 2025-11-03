"""
Core interfaces for Project Argus components.
"""

from .detection import IDetectionPipeline, IModelManager, ISensorFusion, ITamperDetector
from .tracking import IMultiCameraTracker, IReIDMatcher, IBehaviorAnalyzer
from .alerts import IAlertEngine, IEscalationService, INotificationService
from .incidents import IIncidentManager
from .evidence import IEvidenceStore, IForensicsEngine, IAuditLogger
from .health import IHealthMonitor
from .storage import IRepository, ICacheManager
from .security import ISecurityManager, IKeyManager, IAccessController

__all__ = [
    # Detection interfaces
    'IDetectionPipeline', 'IModelManager', 'ISensorFusion', 'ITamperDetector',
    
    # Tracking interfaces
    'IMultiCameraTracker', 'IReIDMatcher', 'IBehaviorAnalyzer',
    
    # Alert interfaces
    'IAlertEngine', 'IEscalationService', 'INotificationService',
    
    # Incident interfaces
    'IIncidentManager',
    
    # Evidence interfaces
    'IEvidenceStore', 'IForensicsEngine', 'IAuditLogger',
    
    # Health interfaces
    'IHealthMonitor',
    
    # Storage interfaces
    'IRepository', 'ICacheManager',
    
    # Security interfaces
    'ISecurityManager', 'IKeyManager', 'IAccessController'
]