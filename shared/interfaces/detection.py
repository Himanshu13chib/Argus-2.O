"""
Detection pipeline interfaces for Project Argus.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

from ..models import Detection, DetectionResult, HealthStatus, VirtualLine


class IDetectionPipeline(ABC):
    """Interface for the main detection pipeline."""
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray, camera_id: str) -> DetectionResult:
        """Process a single frame and return detection results."""
        pass
    
    @abstractmethod
    def update_virtual_lines(self, lines: List[VirtualLine]) -> bool:
        """Update virtual line configurations for crossing detection."""
        pass
    
    @abstractmethod
    def get_health_status(self):
        """Get current health status of the detection pipeline."""
        pass
    
    @abstractmethod
    def start(self) -> None:
        """Start the detection pipeline."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the detection pipeline."""
        pass


class IModelManager(ABC):
    """Interface for managing AI models."""
    
    @abstractmethod
    def load_model(self, model_path: str, model_type: str) -> bool:
        """Load an AI model from file."""
        pass
    
    @abstractmethod
    def optimize_model(self, model_name: str, target_device: str) -> bool:
        """Optimize model for specific hardware (TensorRT, OpenVINO, etc.)."""
        pass
    
    @abstractmethod
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a loaded model."""
        pass
    
    @abstractmethod
    def update_model(self, model_name: str, new_model_path: str) -> bool:
        """Update an existing model with a new version."""
        pass
    
    @abstractmethod
    def validate_model(self, model_name: str) -> bool:
        """Validate model integrity and performance."""
        pass


class ISensorFusion(ABC):
    """Interface for multi-modal sensor fusion."""
    
    @abstractmethod
    def fuse_detections(self, visible: List[Detection], thermal: List[Detection], 
                       radar_data: Optional[Dict[str, Any]] = None) -> List[Detection]:
        """Fuse detections from multiple sensor modalities."""
        pass
    
    @abstractmethod
    def adapt_to_conditions(self, lighting: float, weather: str, temperature: float) -> None:
        """Adapt fusion parameters based on environmental conditions."""
        pass
    
    @abstractmethod
    def get_active_sensors(self) -> List[str]:
        """Get list of currently active sensor types."""
        pass
    
    @abstractmethod
    def calibrate_sensors(self, calibration_data: Dict[str, Any]) -> bool:
        """Calibrate sensor alignment and fusion parameters."""
        pass


class ITamperDetector(ABC):
    """Interface for camera tamper detection."""
    
    @abstractmethod
    def detect_occlusion(self, frame: np.ndarray, camera_id: str) -> bool:
        """Detect if camera lens is occluded."""
        pass
    
    @abstractmethod
    def detect_movement(self, frame: np.ndarray, camera_id: str) -> bool:
        """Detect if camera has been physically moved."""
        pass
    
    @abstractmethod
    def detect_defocus(self, frame: np.ndarray, camera_id: str) -> bool:
        """Detect if camera is out of focus (tampered)."""
        pass
    
    @abstractmethod
    def get_baseline_image(self, camera_id: str) -> Optional[np.ndarray]:
        """Get baseline image for comparison."""
        pass
    
    @abstractmethod
    def update_baseline(self, camera_id: str, frame: np.ndarray) -> None:
        """Update baseline image for tamper detection."""
        pass