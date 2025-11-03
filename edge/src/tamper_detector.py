"""
Camera tamper detection system for Project Argus.

This module implements comprehensive tamper detection including:
- Lens occlusion detection using image analysis
- Camera movement and tilt detection algorithms
- Cable cut and power failure monitoring
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import psutil
import socket

from shared.models.health import ComponentHealth, ComponentStatus, ComponentType


class TamperType(Enum):
    """Types of tampering that can be detected."""
    LENS_OCCLUSION = "lens_occlusion"
    CAMERA_MOVEMENT = "camera_movement"
    CABLE_CUT = "cable_cut"
    POWER_FAILURE = "power_failure"
    SIGNAL_LOSS = "signal_loss"
    LIGHTING_CHANGE = "lighting_change"


@dataclass
class TamperEvent:
    """Represents a detected tamper event."""
    tamper_type: TamperType
    camera_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0
    severity: str = "medium"  # low, medium, high, critical
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'tamper_type': self.tamper_type.value,
            'camera_id': self.camera_id,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'severity': self.severity,
            'description': self.description,
            'metadata': self.metadata
        }


@dataclass
class CameraBaseline:
    """Baseline characteristics of a camera for tamper detection."""
    camera_id: str
    reference_frame: Optional[np.ndarray] = None
    reference_histogram: Optional[np.ndarray] = None
    reference_edges: Optional[np.ndarray] = None
    reference_keypoints: Optional[List] = None
    reference_descriptors: Optional[np.ndarray] = None
    
    # Statistical baselines
    avg_brightness: float = 0.0
    avg_contrast: float = 0.0
    avg_sharpness: float = 0.0
    
    # Movement detection baselines
    reference_corners: Optional[np.ndarray] = None
    homography_threshold: float = 50.0
    
    # Timing baselines
    established_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def is_valid(self) -> bool:
        """Check if baseline is properly established."""
        return (self.reference_frame is not None and 
                self.reference_histogram is not None and
                self.reference_edges is not None)


class TamperDetector:
    """
    Comprehensive tamper detection system for cameras.
    
    Implements multiple detection algorithms:
    - Occlusion detection through brightness and edge analysis
    - Movement detection using feature matching and homography
    - Power/cable monitoring through system metrics
    """
    
    def __init__(self, camera_id: str, baseline_frames: int = 30):
        """
        Initialize tamper detector for a specific camera.
        
        Args:
            camera_id: Unique identifier for the camera
            baseline_frames: Number of frames to use for establishing baseline
        """
        self.camera_id = camera_id
        self.baseline_frames = baseline_frames
        self.logger = logging.getLogger(f"TamperDetector-{camera_id}")
        
        # Baseline data
        self.baseline = CameraBaseline(camera_id=camera_id)
        self.baseline_established = False
        self.baseline_frame_count = 0
        
        # Detection parameters
        self.occlusion_threshold = 0.3  # Brightness reduction threshold
        self.movement_threshold = 50.0  # Pixel displacement threshold
        self.edge_threshold = 0.4  # Edge density reduction threshold
        self.histogram_threshold = 0.6  # Histogram correlation threshold
        
        # Feature detector for movement detection
        self.feature_detector = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Monitoring state
        self.last_frame_time = datetime.now()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Recent events for deduplication
        self.recent_events: List[TamperEvent] = []
        self.event_cooldown = timedelta(minutes=5)
        
    def establish_baseline(self, frame: np.ndarray) -> bool:
        """
        Establish baseline characteristics from incoming frames.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            True if baseline is fully established, False otherwise
        """
        if self.baseline_established:
            return True
            
        with self.lock:
            try:
                if frame is None or frame.size == 0:
                    return False
                
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                
                # Accumulate baseline data
                if self.baseline.reference_frame is None:
                    self.baseline.reference_frame = gray.copy()
                    self.baseline.reference_histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    self.baseline.reference_edges = cv2.Canny(gray, 50, 150)
                    
                    # Extract keypoints and descriptors
                    keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
                    self.baseline.reference_keypoints = keypoints
                    self.baseline.reference_descriptors = descriptors
                    
                    # Calculate statistical measures
                    self.baseline.avg_brightness = np.mean(gray)
                    self.baseline.avg_contrast = np.std(gray)
                    self.baseline.avg_sharpness = self._calculate_sharpness(gray)
                    
                    # Detect corners for movement detection
                    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, 
                                                    qualityLevel=0.01, minDistance=10)
                    self.baseline.reference_corners = corners
                    
                else:
                    # Update baseline with running average
                    alpha = 0.1  # Learning rate
                    self.baseline.reference_frame = cv2.addWeighted(
                        self.baseline.reference_frame, 1-alpha, gray, alpha, 0
                    )
                    
                    # Update statistical measures
                    current_brightness = np.mean(gray)
                    current_contrast = np.std(gray)
                    current_sharpness = self._calculate_sharpness(gray)
                    
                    self.baseline.avg_brightness = (1-alpha) * self.baseline.avg_brightness + alpha * current_brightness
                    self.baseline.avg_contrast = (1-alpha) * self.baseline.avg_contrast + alpha * current_contrast
                    self.baseline.avg_sharpness = (1-alpha) * self.baseline.avg_sharpness + alpha * current_sharpness
                
                self.baseline_frame_count += 1
                
                # Check if baseline is established
                if self.baseline_frame_count >= self.baseline_frames:
                    self.baseline_established = True
                    self.baseline.last_updated = datetime.now()
                    self.logger.info(f"Baseline established for camera {self.camera_id}")
                    return True
                    
                return False
                
            except Exception as e:
                self.logger.error(f"Error establishing baseline: {e}")
                return False
    
    def detect_tampering(self, frame: np.ndarray) -> List[TamperEvent]:
        """
        Detect various types of tampering in the current frame.
        
        Args:
            frame: Current frame from camera
            
        Returns:
            List of detected tamper events
        """
        events = []
        
        if not self.baseline_established:
            if not self.establish_baseline(frame):
                return events
        
        try:
            # Update frame timing
            current_time = datetime.now()
            self.last_frame_time = current_time
            self.consecutive_failures = 0
            
            if frame is None or frame.size == 0:
                events.append(self._create_signal_loss_event())
                return events
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            
            # Run detection algorithms
            events.extend(self._detect_occlusion(gray))
            events.extend(self._detect_movement(gray))
            events.extend(self._detect_lighting_changes(gray))
            
            # Filter recent duplicate events
            events = self._filter_duplicate_events(events)
            
            # Add events to recent history
            self.recent_events.extend(events)
            self._cleanup_old_events()
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error in tamper detection: {e}")
            return []
    
    def detect_system_tampering(self) -> List[TamperEvent]:
        """
        Detect system-level tampering (power, network, hardware).
        
        Returns:
            List of detected system tamper events
        """
        events = []
        
        try:
            # Check for signal loss (no frames received)
            if datetime.now() - self.last_frame_time > timedelta(seconds=10):
                if self.consecutive_failures < self.max_consecutive_failures:
                    self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_consecutive_failures:
                    events.append(self._create_signal_loss_event())
            
            # Check power and system health
            events.extend(self._detect_power_issues())
            events.extend(self._detect_network_issues())
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error in system tamper detection: {e}")
            return []
    
    def _detect_occlusion(self, gray: np.ndarray) -> List[TamperEvent]:
        """Detect lens occlusion through brightness and edge analysis."""
        events = []
        
        try:
            # Calculate current frame statistics
            current_brightness = np.mean(gray)
            current_edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(current_edges > 0) / current_edges.size
            
            # Calculate baseline edge density
            baseline_edge_density = np.sum(self.baseline.reference_edges > 0) / self.baseline.reference_edges.size
            
            # Check for significant brightness reduction
            brightness_ratio = current_brightness / max(self.baseline.avg_brightness, 1)
            edge_ratio = edge_density / max(baseline_edge_density, 0.001)
            
            # Detect occlusion
            if brightness_ratio < self.occlusion_threshold or edge_ratio < self.edge_threshold:
                confidence = 1.0 - min(brightness_ratio, edge_ratio)
                severity = "critical" if confidence > 0.8 else "high" if confidence > 0.6 else "medium"
                
                event = TamperEvent(
                    tamper_type=TamperType.LENS_OCCLUSION,
                    camera_id=self.camera_id,
                    confidence=confidence,
                    severity=severity,
                    description=f"Lens occlusion detected - brightness ratio: {brightness_ratio:.2f}, edge ratio: {edge_ratio:.2f}",
                    metadata={
                        'brightness_ratio': brightness_ratio,
                        'edge_ratio': edge_ratio,
                        'current_brightness': current_brightness,
                        'baseline_brightness': self.baseline.avg_brightness
                    }
                )
                events.append(event)
                
        except Exception as e:
            self.logger.error(f"Error in occlusion detection: {e}")
            
        return events
    
    def _detect_movement(self, gray: np.ndarray) -> List[TamperEvent]:
        """Detect camera movement using feature matching and homography."""
        events = []
        
        try:
            if self.baseline.reference_descriptors is None:
                return events
            
            # Extract features from current frame
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            
            if descriptors is None or len(descriptors) < 10:
                return events
            
            # Match features with baseline
            matches = self.matcher.match(self.baseline.reference_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) < 10:
                # Insufficient matches might indicate significant movement
                event = TamperEvent(
                    tamper_type=TamperType.CAMERA_MOVEMENT,
                    camera_id=self.camera_id,
                    confidence=0.7,
                    severity="high",
                    description=f"Insufficient feature matches detected: {len(matches)}",
                    metadata={'match_count': len(matches)}
                )
                events.append(event)
                return events
            
            # Extract matched points
            src_pts = np.float32([self.baseline.reference_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Calculate homography
            homography, mask = cv2.findHomography(src_pts, dst_pts, 
                                                cv2.RANSAC, 5.0)
            
            if homography is not None:
                # Calculate transformation magnitude
                corners = np.float32([[0, 0], [gray.shape[1], 0], 
                                    [gray.shape[1], gray.shape[0]], [0, gray.shape[0]]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(corners, homography)
                
                # Calculate displacement
                displacement = np.mean(np.linalg.norm(transformed_corners - corners, axis=2))
                
                if displacement > self.movement_threshold:
                    confidence = min(displacement / (self.movement_threshold * 3), 1.0)
                    severity = "critical" if displacement > 100 else "high" if displacement > 75 else "medium"
                    
                    event = TamperEvent(
                        tamper_type=TamperType.CAMERA_MOVEMENT,
                        camera_id=self.camera_id,
                        confidence=confidence,
                        severity=severity,
                        description=f"Camera movement detected - displacement: {displacement:.1f} pixels",
                        metadata={
                            'displacement': displacement,
                            'match_count': len(matches),
                            'inlier_ratio': np.sum(mask) / len(mask) if mask is not None else 0
                        }
                    )
                    events.append(event)
                    
        except Exception as e:
            self.logger.error(f"Error in movement detection: {e}")
            
        return events
    
    def _detect_lighting_changes(self, gray: np.ndarray) -> List[TamperEvent]:
        """Detect significant lighting changes that might indicate tampering."""
        events = []
        
        try:
            # Calculate histogram correlation
            current_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            correlation = cv2.compareHist(self.baseline.reference_histogram, current_hist, cv2.HISTCMP_CORREL)
            
            if correlation < self.histogram_threshold:
                confidence = 1.0 - correlation
                severity = "medium" if confidence > 0.6 else "low"
                
                event = TamperEvent(
                    tamper_type=TamperType.LIGHTING_CHANGE,
                    camera_id=self.camera_id,
                    confidence=confidence,
                    severity=severity,
                    description=f"Significant lighting change detected - correlation: {correlation:.3f}",
                    metadata={
                        'histogram_correlation': correlation,
                        'current_brightness': np.mean(gray),
                        'baseline_brightness': self.baseline.avg_brightness
                    }
                )
                events.append(event)
                
        except Exception as e:
            self.logger.error(f"Error in lighting change detection: {e}")
            
        return events
    
    def _detect_power_issues(self) -> List[TamperEvent]:
        """Detect power-related issues."""
        events = []
        
        try:
            # Check system power status
            battery = psutil.sensors_battery()
            if battery:
                if not battery.power_plugged and battery.percent < 20:
                    event = TamperEvent(
                        tamper_type=TamperType.POWER_FAILURE,
                        camera_id=self.camera_id,
                        confidence=0.9,
                        severity="critical",
                        description=f"Low battery power detected: {battery.percent}%",
                        metadata={'battery_percent': battery.percent, 'power_plugged': battery.power_plugged}
                    )
                    events.append(event)
            
            # Check CPU temperature (if available)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current > 80:  # High temperature threshold
                                event = TamperEvent(
                                    tamper_type=TamperType.POWER_FAILURE,
                                    camera_id=self.camera_id,
                                    confidence=0.7,
                                    severity="high",
                                    description=f"High system temperature: {entry.current}°C",
                                    metadata={'temperature': entry.current, 'sensor': name}
                                )
                                events.append(event)
            except AttributeError:
                # sensors_temperatures not available on this platform
                pass
                            
        except Exception as e:
            self.logger.error(f"Error in power issue detection: {e}")
            
        return events
    
    def _detect_network_issues(self) -> List[TamperEvent]:
        """Detect network connectivity issues."""
        events = []
        
        try:
            # Check network connectivity
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=3)
            except OSError:
                event = TamperEvent(
                    tamper_type=TamperType.CABLE_CUT,
                    camera_id=self.camera_id,
                    confidence=0.8,
                    severity="high",
                    description="Network connectivity lost",
                    metadata={'connectivity_test': 'failed'}
                )
                events.append(event)
                
        except Exception as e:
            self.logger.error(f"Error in network issue detection: {e}")
            
        return events
    
    def _create_signal_loss_event(self) -> TamperEvent:
        """Create a signal loss tamper event."""
        return TamperEvent(
            tamper_type=TamperType.SIGNAL_LOSS,
            camera_id=self.camera_id,
            confidence=0.9,
            severity="critical",
            description=f"Signal loss detected - no frames for {self.consecutive_failures} checks",
            metadata={'consecutive_failures': self.consecutive_failures}
        )
    
    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _filter_duplicate_events(self, events: List[TamperEvent]) -> List[TamperEvent]:
        """Filter out duplicate events that occurred recently."""
        filtered_events = []
        
        for event in events:
            is_duplicate = False
            for recent_event in self.recent_events:
                if (recent_event.tamper_type == event.tamper_type and
                    recent_event.camera_id == event.camera_id and
                    datetime.now() - recent_event.timestamp < self.event_cooldown):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_events.append(event)
                
        return filtered_events
    
    def _cleanup_old_events(self) -> None:
        """Remove old events from recent history."""
        cutoff_time = datetime.now() - self.event_cooldown * 2
        self.recent_events = [event for event in self.recent_events 
                            if event.timestamp > cutoff_time]
    
    def update_baseline(self, frame: np.ndarray) -> None:
        """Update baseline with new frame (for adaptive baseline)."""
        if not self.baseline_established:
            return
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            
            # Gradual baseline update
            alpha = 0.01  # Very slow learning rate for stability
            self.baseline.reference_frame = cv2.addWeighted(
                self.baseline.reference_frame, 1-alpha, gray, alpha, 0
            )
            
            # Update statistical measures
            current_brightness = np.mean(gray)
            current_contrast = np.std(gray)
            current_sharpness = self._calculate_sharpness(gray)
            
            self.baseline.avg_brightness = (1-alpha) * self.baseline.avg_brightness + alpha * current_brightness
            self.baseline.avg_contrast = (1-alpha) * self.baseline.avg_contrast + alpha * current_contrast
            self.baseline.avg_sharpness = (1-alpha) * self.baseline.avg_sharpness + alpha * current_sharpness
            
            self.baseline.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating baseline: {e}")
    
    def _point_in_polygon(self, x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.
        
        Args:
            x: X coordinate of point
            y: Y coordinate of point
            polygon: List of (x, y) tuples defining polygon vertices
            
        Returns:
            True if point is inside polygon, False otherwise
        """
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_detection_status(self) -> Dict[str, Any]:
        """Get current status of tamper detection system."""
        return {
            'camera_id': self.camera_id,
            'baseline_established': self.baseline_established,
            'baseline_frame_count': self.baseline_frame_count,
            'last_frame_time': self.last_frame_time.isoformat(),
            'consecutive_failures': self.consecutive_failures,
            'recent_events_count': len(self.recent_events),
            'baseline_stats': {
                'avg_brightness': self.baseline.avg_brightness,
                'avg_contrast': self.baseline.avg_contrast,
                'avg_sharpness': self.baseline.avg_sharpness,
                'established_at': self.baseline.established_at.isoformat() if self.baseline_established else None,
                'last_updated': self.baseline.last_updated.isoformat() if self.baseline_established else None
            }
        }