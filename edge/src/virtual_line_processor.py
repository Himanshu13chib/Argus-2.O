"""
Virtual Line Detection Engine for Project Argus.

Implements configurable boundary detection with polygon and line-based crossing
detection algorithms, real-time virtual line overlay, and configuration management
for border crossing detection (Requirements 1.3, 4.2).
"""

import logging
import time
import numpy as np
import cv2
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
import json
import threading

from shared.models.virtual_line import VirtualLine, Point, VirtualLineType, CrossingDirection
from shared.models.detection import Detection, BoundingBox


logger = logging.getLogger(__name__)


class CrossingEvent(Enum):
    """Types of crossing events."""
    ENTRY = "entry"
    EXIT = "exit"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class CrossingDetection:
    """Result of virtual line crossing detection."""
    detection_id: str
    virtual_line_id: str
    crossing_point: Point
    crossing_type: CrossingEvent
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryPoint:
    """Point in a person's trajectory with timestamp."""
    point: Point
    timestamp: datetime
    detection_id: str


class VirtualLineProcessor:
    """
    Virtual line detection engine with configurable boundary detection.
    
    Supports polygon and line-based crossing detection algorithms with
    real-time overlay and configuration management (Requirements 1.3, 4.2).
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.virtual_lines: Dict[str, VirtualLine] = {}
        self.trajectories: Dict[str, List[TrajectoryPoint]] = {}
        self.crossing_history: List[CrossingDetection] = []
        self._lock = threading.Lock()
        
        # Configuration parameters
        self.max_trajectory_length = config.get("max_trajectory_length", 10)
        self.trajectory_timeout_seconds = config.get("trajectory_timeout_seconds", 30)
        self.crossing_cooldown_seconds = config.get("crossing_cooldown_seconds", 5)
        self.line_thickness_pixels = config.get("line_thickness_pixels", 20)
        self.polygon_edge_tolerance = config.get("polygon_edge_tolerance", 10)
        
        # Performance tracking
        self.processing_times = []
        self.total_crossings_detected = 0
        self.false_positive_count = 0
        
        logger.info("VirtualLineProcessor initialized")
        logger.info(f"Max trajectory length: {self.max_trajectory_length}")
        logger.info(f"Trajectory timeout: {self.trajectory_timeout_seconds}s")
    
    def add_virtual_line(self, virtual_line: VirtualLine) -> bool:
        """
        Add a new virtual line configuration.
        
        Args:
            virtual_line: VirtualLine configuration to add
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        try:
            with self._lock:
                # Validate virtual line
                if not self._validate_virtual_line(virtual_line):
                    logger.error(f"Invalid virtual line configuration: {virtual_line.id}")
                    return False
                
                self.virtual_lines[virtual_line.id] = virtual_line
                
                logger.info(f"Added virtual line '{virtual_line.id}' for camera {virtual_line.camera_id}")
                logger.info(f"Type: {virtual_line.type.value}, Points: {len(virtual_line.points)}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error adding virtual line {virtual_line.id}: {e}")
            return False
    
    def remove_virtual_line(self, line_id: str) -> bool:
        """Remove a virtual line configuration."""
        try:
            with self._lock:
                if line_id in self.virtual_lines:
                    del self.virtual_lines[line_id]
                    logger.info(f"Removed virtual line: {line_id}")
                    return True
                else:
                    logger.warning(f"Virtual line not found: {line_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error removing virtual line {line_id}: {e}")
            return False
    
    def update_virtual_line(self, virtual_line: VirtualLine) -> bool:
        """Update an existing virtual line configuration."""
        try:
            with self._lock:
                if virtual_line.id not in self.virtual_lines:
                    logger.warning(f"Virtual line not found for update: {virtual_line.id}")
                    return False
                
                if not self._validate_virtual_line(virtual_line):
                    logger.error(f"Invalid virtual line configuration: {virtual_line.id}")
                    return False
                
                self.virtual_lines[virtual_line.id] = virtual_line
                logger.info(f"Updated virtual line: {virtual_line.id}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating virtual line {virtual_line.id}: {e}")
            return False
    
    def get_virtual_lines(self, camera_id: Optional[str] = None) -> List[VirtualLine]:
        """Get virtual lines, optionally filtered by camera ID."""
        with self._lock:
            if camera_id:
                return [vl for vl in self.virtual_lines.values() if vl.camera_id == camera_id]
            else:
                return list(self.virtual_lines.values())
    
    def process_detections(self, detections: List[Detection], camera_id: str) -> List[CrossingDetection]:
        """
        Process detections for virtual line crossings.
        
        Args:
            detections: List of person detections to check
            camera_id: Camera identifier
            
        Returns:
            List of crossing detections
        """
        start_time = time.perf_counter()
        
        try:
            crossing_detections = []
            
            # Get virtual lines for this camera
            camera_lines = self.get_virtual_lines(camera_id)
            if not camera_lines:
                return crossing_detections
            
            # Update trajectories with new detections
            self._update_trajectories(detections, camera_id)
            
            # Check each detection against virtual lines
            for detection in detections:
                for virtual_line in camera_lines:
                    # Skip inactive lines
                    if not virtual_line.active:
                        continue
                    
                    # Skip if line is not active at current time
                    if not virtual_line.is_active_at_time(datetime.now()):
                        continue
                    
                    # Check for crossing
                    crossing = self._check_detection_crossing(detection, virtual_line)
                    if crossing:
                        crossing_detections.append(crossing)
                        
                        with self._lock:
                            self.total_crossings_detected += 1
                            self.crossing_history.append(crossing)
                            
                            # Limit crossing history size
                            if len(self.crossing_history) > 1000:
                                self.crossing_history = self.crossing_history[-500:]
                        
                        logger.info(f"CROSSING DETECTED: {crossing.detection_id} crossed {crossing.virtual_line_id}")
            
            # Clean up old trajectories
            self._cleanup_old_trajectories()
            
            # Update performance tracking
            processing_time = (time.perf_counter() - start_time) * 1000
            with self._lock:
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
            
            return crossing_detections
            
        except Exception as e:
            logger.error(f"Error processing detections for virtual lines: {e}")
            return []
    
    def _validate_virtual_line(self, virtual_line: VirtualLine) -> bool:
        """Validate virtual line configuration."""
        try:
            # Check required fields
            if not virtual_line.id or not virtual_line.camera_id:
                return False
            
            # Check points
            if not virtual_line.points or len(virtual_line.points) < 2:
                logger.error(f"Virtual line {virtual_line.id} needs at least 2 points")
                return False
            
            # Validate point coordinates
            for point in virtual_line.points:
                if not isinstance(point.x, (int, float)) or not isinstance(point.y, (int, float)):
                    logger.error(f"Invalid point coordinates in virtual line {virtual_line.id}")
                    return False
                
                if point.x < 0 or point.y < 0:
                    logger.warning(f"Negative coordinates in virtual line {virtual_line.id}")
            
            # Type-specific validation
            if virtual_line.type == VirtualLineType.POLYGON and len(virtual_line.points) < 3:
                logger.error(f"Polygon virtual line {virtual_line.id} needs at least 3 points")
                return False
            
            # Validate sensitivity
            if not (0.0 <= virtual_line.sensitivity <= 1.0):
                logger.error(f"Invalid sensitivity {virtual_line.sensitivity} for virtual line {virtual_line.id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating virtual line: {e}")
            return False
    
    def _update_trajectories(self, detections: List[Detection], camera_id: str) -> None:
        """Update person trajectories with new detections."""
        current_time = datetime.now()
        
        with self._lock:
            for detection in detections:
                # Calculate person foot position (bottom center of bounding box)
                foot_x = detection.bbox.x + detection.bbox.width / 2
                foot_y = detection.bbox.y + detection.bbox.height
                
                trajectory_point = TrajectoryPoint(
                    point=Point(x=foot_x, y=foot_y),
                    timestamp=current_time,
                    detection_id=detection.id
                )
                
                # Use a simplified tracking ID (in production, this would use proper tracking)
                track_id = f"{camera_id}_{int(foot_x/50)}_{int(foot_y/50)}"
                
                if track_id not in self.trajectories:
                    self.trajectories[track_id] = []
                
                self.trajectories[track_id].append(trajectory_point)
                
                # Limit trajectory length
                if len(self.trajectories[track_id]) > self.max_trajectory_length:
                    self.trajectories[track_id] = self.trajectories[track_id][-self.max_trajectory_length:]
    
    def _cleanup_old_trajectories(self) -> None:
        """Remove old trajectories that have timed out."""
        current_time = datetime.now()
        timeout_threshold = current_time.timestamp() - self.trajectory_timeout_seconds
        
        with self._lock:
            tracks_to_remove = []
            
            for track_id, trajectory in self.trajectories.items():
                if not trajectory:
                    tracks_to_remove.append(track_id)
                    continue
                
                # Check if latest point is too old
                latest_point = trajectory[-1]
                if latest_point.timestamp.timestamp() < timeout_threshold:
                    tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                del self.trajectories[track_id]
            
            if tracks_to_remove:
                logger.debug(f"Cleaned up {len(tracks_to_remove)} old trajectories")
    
    def _check_detection_crossing(self, detection: Detection, virtual_line: VirtualLine) -> Optional[CrossingDetection]:
        """
        Check if a detection crosses a virtual line.
        
        Args:
            detection: Person detection to check
            virtual_line: Virtual line configuration
            
        Returns:
            CrossingDetection if crossing detected, None otherwise
        """
        try:
            # Calculate person foot position
            foot_x = detection.bbox.x + detection.bbox.width / 2
            foot_y = detection.bbox.y + detection.bbox.height
            foot_point = Point(x=foot_x, y=foot_y)
            
            # Check crossing based on virtual line type
            if virtual_line.type == VirtualLineType.LINE:
                crossing_point = self._check_line_crossing(foot_point, virtual_line, detection.id)
            elif virtual_line.type == VirtualLineType.POLYGON:
                crossing_point = self._check_polygon_crossing(foot_point, virtual_line, detection.id)
            else:
                logger.warning(f"Unsupported virtual line type: {virtual_line.type}")
                return None
            
            if crossing_point:
                # Calculate crossing confidence based on detection confidence and line sensitivity
                base_confidence = detection.confidence
                line_sensitivity = virtual_line.sensitivity
                crossing_confidence = base_confidence * line_sensitivity
                
                # Apply distance-based confidence adjustment
                distance_factor = self._calculate_distance_confidence_factor(foot_point, virtual_line)
                final_confidence = crossing_confidence * distance_factor
                
                return CrossingDetection(
                    detection_id=detection.id,
                    virtual_line_id=virtual_line.id,
                    crossing_point=crossing_point,
                    crossing_type=CrossingEvent.BIDIRECTIONAL,  # Simplified for now
                    confidence=final_confidence,
                    timestamp=datetime.now(),
                    metadata={
                        "detection_bbox": detection.bbox.to_dict(),
                        "foot_position": {"x": foot_x, "y": foot_y},
                        "line_type": virtual_line.type.value,
                        "line_sensitivity": virtual_line.sensitivity,
                        "distance_factor": distance_factor,
                        "original_confidence": detection.confidence
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking detection crossing: {e}")
            return None
    
    def _check_line_crossing(self, point: Point, virtual_line: VirtualLine, detection_id: str) -> Optional[Point]:
        """
        Check if a point crosses a line-based virtual line.
        
        Args:
            point: Point to check (person foot position)
            virtual_line: Line-based virtual line
            detection_id: Detection identifier for trajectory lookup
            
        Returns:
            Crossing point if detected, None otherwise
        """
        if len(virtual_line.points) < 2:
            return None
        
        # Get line segments
        line_segments = virtual_line.get_line_segments()
        
        for line_start, line_end in line_segments:
            # Calculate distance from point to line segment
            distance = self._point_to_line_distance(point, line_start, line_end)
            
            # Dynamic threshold based on line sensitivity and thickness
            threshold = self.line_thickness_pixels * (1.0 + (1.0 - virtual_line.sensitivity))
            
            if distance <= threshold:
                # Check if this is a new crossing (not just hovering near line)
                if self._is_new_crossing(point, virtual_line, detection_id):
                    # Return the closest point on the line as crossing point
                    return self._closest_point_on_line(point, line_start, line_end)
        
        return None
    
    def _check_polygon_crossing(self, point: Point, virtual_line: VirtualLine, detection_id: str) -> Optional[Point]:
        """
        Check if a point crosses into/out of a polygon-based virtual line.
        
        Args:
            point: Point to check
            virtual_line: Polygon-based virtual line
            detection_id: Detection identifier for trajectory lookup
            
        Returns:
            Crossing point if detected, None otherwise
        """
        if len(virtual_line.points) < 3:
            return None
        
        # Check if point is inside polygon
        is_inside = virtual_line.point_in_polygon(point)
        
        # Look up previous position to detect crossing
        previous_inside = self._get_previous_polygon_state(point, virtual_line, detection_id)
        
        # Crossing detected if state changed
        if previous_inside is not None and previous_inside != is_inside:
            # Store current state for next check
            self._store_polygon_state(point, virtual_line, detection_id, is_inside)
            
            return point  # Return the crossing point
        
        # Store current state for next check
        self._store_polygon_state(point, virtual_line, detection_id, is_inside)
        
        return None
    
    def _point_to_line_distance(self, point: Point, line_start: Point, line_end: Point) -> float:
        """Calculate perpendicular distance from point to line segment."""
        # Vector from line start to point
        px = point.x - line_start.x
        py = point.y - line_start.y
        
        # Vector of line segment
        lx = line_end.x - line_start.x
        ly = line_end.y - line_start.y
        
        # Length squared of line segment
        length_sq = lx * lx + ly * ly
        
        if length_sq == 0:
            # Line segment is a point
            return np.sqrt(px * px + py * py)
        
        # Parameter t represents position along line segment (0 to 1)
        t = max(0, min(1, (px * lx + py * ly) / length_sq))
        
        # Closest point on line segment
        closest_x = line_start.x + t * lx
        closest_y = line_start.y + t * ly
        
        # Distance from point to closest point on line
        return np.sqrt((point.x - closest_x) ** 2 + (point.y - closest_y) ** 2)
    
    def _closest_point_on_line(self, point: Point, line_start: Point, line_end: Point) -> Point:
        """Find the closest point on a line segment to a given point."""
        # Vector from line start to point
        px = point.x - line_start.x
        py = point.y - line_start.y
        
        # Vector of line segment
        lx = line_end.x - line_start.x
        ly = line_end.y - line_start.y
        
        # Length squared of line segment
        length_sq = lx * lx + ly * ly
        
        if length_sq == 0:
            return Point(x=line_start.x, y=line_start.y)
        
        # Parameter t represents position along line segment
        t = max(0, min(1, (px * lx + py * ly) / length_sq))
        
        # Closest point on line segment
        return Point(
            x=line_start.x + t * lx,
            y=line_start.y + t * ly
        )
    
    def _is_new_crossing(self, point: Point, virtual_line: VirtualLine, detection_id: str) -> bool:
        """
        Check if this is a new crossing or just hovering near the line.
        
        This is a simplified implementation. In production, this would use
        proper trajectory analysis and crossing cooldown.
        """
        # Check crossing cooldown
        current_time = datetime.now()
        cooldown_threshold = current_time.timestamp() - self.crossing_cooldown_seconds
        
        with self._lock:
            # Check recent crossings for this detection/line combination
            recent_crossings = [
                c for c in self.crossing_history
                if (c.detection_id == detection_id and 
                    c.virtual_line_id == virtual_line.id and
                    c.timestamp.timestamp() > cooldown_threshold)
            ]
            
            return len(recent_crossings) == 0
    
    def _get_previous_polygon_state(self, point: Point, virtual_line: VirtualLine, detection_id: str) -> Optional[bool]:
        """Get previous inside/outside state for polygon crossing detection."""
        # This is a simplified implementation
        # In production, this would use proper trajectory tracking
        state_key = f"{detection_id}_{virtual_line.id}_polygon_state"
        return getattr(self, '_polygon_states', {}).get(state_key)
    
    def _store_polygon_state(self, point: Point, virtual_line: VirtualLine, detection_id: str, is_inside: bool) -> None:
        """Store current inside/outside state for polygon crossing detection."""
        if not hasattr(self, '_polygon_states'):
            self._polygon_states = {}
        
        state_key = f"{detection_id}_{virtual_line.id}_polygon_state"
        self._polygon_states[state_key] = is_inside
        
        # Clean up old states (keep only recent ones)
        if len(self._polygon_states) > 1000:
            # Keep only the most recent 500 states
            keys = list(self._polygon_states.keys())
            for key in keys[:-500]:
                del self._polygon_states[key]
    
    def _calculate_distance_confidence_factor(self, point: Point, virtual_line: VirtualLine) -> float:
        """
        Calculate confidence factor based on distance to virtual line.
        
        Closer to the line = higher confidence.
        """
        if virtual_line.type == VirtualLineType.LINE:
            # Find minimum distance to any line segment
            min_distance = float('inf')
            
            for line_start, line_end in virtual_line.get_line_segments():
                distance = self._point_to_line_distance(point, line_start, line_end)
                min_distance = min(min_distance, distance)
            
            # Convert distance to confidence factor (0.5 to 1.0)
            max_distance = self.line_thickness_pixels * 2
            if min_distance >= max_distance:
                return 0.5
            else:
                return 0.5 + 0.5 * (1.0 - min_distance / max_distance)
        
        elif virtual_line.type == VirtualLineType.POLYGON:
            # For polygons, confidence is based on how close to the edge
            # This is simplified - in production would calculate actual distance to polygon edge
            return 1.0
        
        return 1.0
    
    def generate_overlay_image(self, frame: np.ndarray, camera_id: str, 
                             show_detections: bool = True) -> np.ndarray:
        """
        Generate real-time virtual line overlay on camera frame.
        
        Args:
            frame: Camera frame to overlay on
            camera_id: Camera identifier
            show_detections: Whether to show detection boxes
            
        Returns:
            Frame with virtual line overlay
        """
        try:
            overlay_frame = frame.copy()
            
            # Get virtual lines for this camera
            camera_lines = self.get_virtual_lines(camera_id)
            
            for virtual_line in camera_lines:
                if not virtual_line.active:
                    continue
                
                # Parse color (hex to BGR)
                color_bgr = self._hex_to_bgr(virtual_line.color)
                thickness = max(1, virtual_line.thickness)
                
                # Draw virtual line based on type
                if virtual_line.type == VirtualLineType.LINE:
                    self._draw_line_overlay(overlay_frame, virtual_line, color_bgr, thickness)
                elif virtual_line.type == VirtualLineType.POLYGON:
                    self._draw_polygon_overlay(overlay_frame, virtual_line, color_bgr, thickness)
                
                # Add line label
                if virtual_line.points:
                    label_point = virtual_line.points[0]
                    label_text = virtual_line.name or f"Line {virtual_line.id[:8]}"
                    cv2.putText(overlay_frame, label_text,
                              (int(label_point.x), int(label_point.y - 10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
            
            # Apply opacity blending
            if camera_lines:
                alpha = 0.7  # Default opacity
                if camera_lines[0].opacity:
                    alpha = camera_lines[0].opacity
                
                overlay_frame = cv2.addWeighted(frame, 1 - alpha, overlay_frame, alpha, 0)
            
            return overlay_frame
            
        except Exception as e:
            logger.error(f"Error generating overlay image: {e}")
            return frame
    
    def _draw_line_overlay(self, frame: np.ndarray, virtual_line: VirtualLine, 
                          color: Tuple[int, int, int], thickness: int) -> None:
        """Draw line-based virtual line overlay."""
        points = virtual_line.points
        
        for i in range(len(points) - 1):
            start_point = (int(points[i].x), int(points[i].y))
            end_point = (int(points[i + 1].x), int(points[i + 1].y))
            
            cv2.line(frame, start_point, end_point, color, thickness)
            
            # Draw direction arrow if specified
            if virtual_line.direction != CrossingDirection.BIDIRECTIONAL:
                self._draw_direction_arrow(frame, start_point, end_point, color, thickness)
    
    def _draw_polygon_overlay(self, frame: np.ndarray, virtual_line: VirtualLine,
                            color: Tuple[int, int, int], thickness: int) -> None:
        """Draw polygon-based virtual line overlay."""
        points = virtual_line.points
        
        if len(points) < 3:
            return
        
        # Convert points to numpy array for OpenCV
        np_points = np.array([(int(p.x), int(p.y)) for p in points], np.int32)
        np_points = np_points.reshape((-1, 1, 2))
        
        # Draw filled polygon with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np_points], color)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        # Draw polygon outline
        cv2.polylines(frame, [np_points], True, color, thickness)
    
    def _draw_direction_arrow(self, frame: np.ndarray, start: Tuple[int, int], 
                            end: Tuple[int, int], color: Tuple[int, int, int], thickness: int) -> None:
        """Draw direction arrow on line."""
        # Calculate arrow position (middle of line)
        mid_x = (start[0] + end[0]) // 2
        mid_y = (start[1] + end[1]) // 2
        
        # Calculate arrow direction
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx * dx + dy * dy)
        
        if length > 0:
            # Normalize direction
            dx /= length
            dy /= length
            
            # Arrow size
            arrow_length = 20
            arrow_angle = 0.5
            
            # Arrow tip
            tip_x = int(mid_x + dx * arrow_length)
            tip_y = int(mid_y + dy * arrow_length)
            
            # Arrow wings
            wing1_x = int(tip_x - dx * arrow_length * 0.7 + dy * arrow_length * 0.3)
            wing1_y = int(tip_y - dy * arrow_length * 0.7 - dx * arrow_length * 0.3)
            
            wing2_x = int(tip_x - dx * arrow_length * 0.7 - dy * arrow_length * 0.3)
            wing2_y = int(tip_y - dy * arrow_length * 0.7 + dx * arrow_length * 0.3)
            
            # Draw arrow
            cv2.line(frame, (mid_x, mid_y), (tip_x, tip_y), color, thickness)
            cv2.line(frame, (tip_x, tip_y), (wing1_x, wing1_y), color, thickness)
            cv2.line(frame, (tip_x, tip_y), (wing2_x, wing2_y), color, thickness)
    
    def _hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to BGR tuple for OpenCV."""
        try:
            # Remove # if present
            hex_color = hex_color.lstrip('#')
            
            # Convert to RGB
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            # Convert RGB to BGR for OpenCV
            return (rgb[2], rgb[1], rgb[0])
            
        except Exception:
            # Default to red if parsing fails
            return (0, 0, 255)
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export virtual line configuration to dictionary."""
        with self._lock:
            return {
                "virtual_lines": [vl.to_dict() for vl in self.virtual_lines.values()],
                "config": self.config,
                "statistics": {
                    "total_lines": len(self.virtual_lines),
                    "total_crossings": self.total_crossings_detected,
                    "avg_processing_time_ms": np.mean(self.processing_times) if self.processing_times else 0
                },
                "export_timestamp": datetime.now().isoformat()
            }
    
    def import_configuration(self, config_data: Dict[str, Any]) -> bool:
        """Import virtual line configuration from dictionary."""
        try:
            with self._lock:
                # Clear existing lines
                self.virtual_lines.clear()
                
                # Import virtual lines
                for line_data in config_data.get("virtual_lines", []):
                    virtual_line = VirtualLine.from_dict(line_data)
                    if self._validate_virtual_line(virtual_line):
                        self.virtual_lines[virtual_line.id] = virtual_line
                    else:
                        logger.warning(f"Skipped invalid virtual line: {virtual_line.id}")
                
                # Update configuration if provided
                if "config" in config_data:
                    self.config.update(config_data["config"])
                
                logger.info(f"Imported {len(self.virtual_lines)} virtual lines")
                return True
                
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics and performance metrics."""
        with self._lock:
            return {
                "total_virtual_lines": len(self.virtual_lines),
                "active_virtual_lines": sum(1 for vl in self.virtual_lines.values() if vl.active),
                "total_crossings_detected": self.total_crossings_detected,
                "false_positive_count": self.false_positive_count,
                "active_trajectories": len(self.trajectories),
                "avg_processing_time_ms": np.mean(self.processing_times) if self.processing_times else 0,
                "max_processing_time_ms": max(self.processing_times) if self.processing_times else 0,
                "recent_crossings": len([c for c in self.crossing_history 
                                       if (datetime.now() - c.timestamp).seconds < 3600]),
                "configuration": {
                    "max_trajectory_length": self.max_trajectory_length,
                    "trajectory_timeout_seconds": self.trajectory_timeout_seconds,
                    "crossing_cooldown_seconds": self.crossing_cooldown_seconds,
                    "line_thickness_pixels": self.line_thickness_pixels
                }
            }