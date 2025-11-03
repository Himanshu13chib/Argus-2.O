"""
Tracking and re-identification interfaces for Project Argus.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from ..models import Detection, Track, GlobalTrack, Trajectory


class IMultiCameraTracker(ABC):
    """Interface for multi-camera tracking system."""
    
    @abstractmethod
    def update_tracks(self, detections: List[Detection], camera_id: str) -> List[Track]:
        """Update tracks with new detections from a camera."""
        pass
    
    @abstractmethod
    def match_across_cameras(self, track: Track) -> Optional[GlobalTrack]:
        """Match a local track to a global track across cameras."""
        pass
    
    @abstractmethod
    def get_trajectory(self, track_id: str) -> Optional[Trajectory]:
        """Get trajectory for a specific track."""
        pass
    
    @abstractmethod
    def get_active_tracks(self, camera_id: Optional[str] = None) -> List[Track]:
        """Get all active tracks, optionally filtered by camera."""
        pass
    
    @abstractmethod
    def cleanup_old_tracks(self, max_age_seconds: int) -> int:
        """Remove old inactive tracks and return count removed."""
        pass


class IReIDMatcher(ABC):
    """Interface for person re-identification."""
    
    @abstractmethod
    def extract_features(self, person_crop: np.ndarray) -> np.ndarray:
        """Extract feature embedding from person image crop."""
        pass
    
    @abstractmethod
    def match_person(self, features: np.ndarray, gallery: List[Tuple[str, np.ndarray]], 
                    threshold: float = 0.7) -> Optional[Tuple[str, float]]:
        """Match person features against gallery, return (person_id, confidence)."""
        pass
    
    @abstractmethod
    def add_to_gallery(self, person_id: str, features: np.ndarray) -> None:
        """Add person features to the gallery."""
        pass
    
    @abstractmethod
    def update_gallery(self, person_id: str, new_features: np.ndarray) -> None:
        """Update existing person features in gallery."""
        pass
    
    @abstractmethod
    def get_gallery_size(self) -> int:
        """Get number of persons in the gallery."""
        pass
    
    @abstractmethod
    def cleanup_gallery(self, max_age_days: int) -> int:
        """Remove old entries from gallery and return count removed."""
        pass


class IBehaviorAnalyzer(ABC):
    """Interface for behavioral analysis and anomaly detection."""
    
    @abstractmethod
    def analyze_loitering(self, trajectory: Trajectory, area_bounds: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """Analyze trajectory for loitering behavior."""
        pass
    
    @abstractmethod
    def detect_anomalous_movement(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Detect anomalous movement patterns."""
        pass
    
    @abstractmethod
    def analyze_group_behavior(self, tracks: List[Track]) -> Dict[str, Any]:
        """Analyze behavior of multiple people moving together."""
        pass
    
    @abstractmethod
    def calculate_risk_score(self, track: Track, context: Dict[str, Any]) -> float:
        """Calculate risk score based on behavior analysis."""
        pass
    
    @abstractmethod
    def get_historical_patterns(self, location: str, time_window_hours: int) -> Dict[str, Any]:
        """Get historical movement patterns for a location."""
        pass
    
    @abstractmethod
    def update_baseline_behavior(self, tracks: List[Track], location: str) -> None:
        """Update baseline behavior patterns for a location."""
        pass