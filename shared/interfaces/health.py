"""
Health monitoring interfaces for Project Argus.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable

from ..models import HealthStatus, ComponentHealth, ComponentType, ComponentStatus


class IHealthMonitor(ABC):
    """Interface for system health monitoring."""
    
    @abstractmethod
    def register_component(self, component_id: str, component_type: ComponentType, 
                          check_interval_seconds: int = 30) -> None:
        """Register a component for health monitoring."""
        pass
    
    @abstractmethod
    def unregister_component(self, component_id: str) -> None:
        """Unregister a component from health monitoring."""
        pass
    
    @abstractmethod
    def update_component_health(self, component_id: str, health_data: Dict[str, Any]) -> None:
        """Update health status for a component."""
        pass
    
    @abstractmethod
    def get_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        """Get health status of specific component."""
        pass
    
    @abstractmethod
    def get_system_health(self) -> HealthStatus:
        """Get overall system health status."""
        pass
    
    @abstractmethod
    def get_unhealthy_components(self) -> List[ComponentHealth]:
        """Get all components that are not healthy."""
        pass
    
    @abstractmethod
    def register_health_check(self, component_id: str, 
                             check_function: Callable[[], Dict[str, Any]]) -> None:
        """Register custom health check function for component."""
        pass
    
    @abstractmethod
    def run_health_checks(self) -> None:
        """Run all registered health checks."""
        pass
    
    @abstractmethod
    def set_alert_thresholds(self, component_type: ComponentType, 
                           thresholds: Dict[str, float]) -> None:
        """Set alert thresholds for component type."""
        pass
    
    @abstractmethod
    def get_health_history(self, component_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health history for a component."""
        pass
    
    @abstractmethod
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get aggregated system performance metrics."""
        pass
    
    @abstractmethod
    def predict_failures(self, component_id: str) -> Dict[str, Any]:
        """Predict potential component failures based on trends."""
        pass