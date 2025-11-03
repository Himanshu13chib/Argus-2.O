"""
Behavioral analysis system for loitering detection, anomaly detection, and risk scoring.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import math
from dataclasses import dataclass
from loguru import logger

from shared.interfaces.tracking import IBehaviorAnalyzer
from shared.models import Track, Trajectory, Point


@dataclass
class MovementPattern:
    """Represents a movement pattern for analysis."""
    velocity_profile: List[float]
    direction_changes: int
    stops_count: int
    total_distance: float
    duration_seconds: float
    area_coverage: float
    
    def get_average_velocity(self) -> float:
        """Get average velocity."""
        return sum(self.velocity_profile) / len(self.velocity_profile) if self.velocity_profile else 0.0
    
    def get_velocity_variance(self) -> float:
        """Get velocity variance."""
        if not self.velocity_profile:
            return 0.0
        avg = self.get_average_velocity()
        return sum((v - avg) ** 2 for v in self.velocity_profile) / len(self.velocity_profile)


@dataclass
class LoiteringAnalysis:
    """Results of loitering analysis."""
    is_loitering: bool
    loitering_score: float
    time_in_area: float
    movement_ratio: float
    stationary_periods: List[Tuple[datetime, datetime]]
    risk_level: str


@dataclass
class AnomalyAnalysis:
    """Results of anomaly detection."""
    is_anomalous: bool
    anomaly_score: float
    anomaly_types: List[str]
    confidence: float
    details: Dict[str, Any]


class BehaviorAnalyzer(IBehaviorAnalyzer):
    """Behavioral analysis system for detecting suspicious activities."""
    
    def __init__(self, 
                 loitering_threshold_seconds: float = 300.0,
                 movement_threshold_pixels: float = 10.0,
                 velocity_anomaly_threshold: float = 2.0,
                 direction_change_threshold: int = 5):
        """
        Initialize behavior analyzer.
        
        Args:
            loitering_threshold_seconds: Time threshold for loitering detection
            movement_threshold_pixels: Minimum movement to not be considered stationary
            velocity_anomaly_threshold: Standard deviations for velocity anomaly
            direction_change_threshold: Threshold for excessive direction changes
        """
        self.loitering_threshold = loitering_threshold_seconds
        self.movement_threshold = movement_threshold_pixels
        self.velocity_anomaly_threshold = velocity_anomaly_threshold
        self.direction_change_threshold = direction_change_threshold
        
        # Historical data for baseline behavior
        self.location_baselines: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.velocity_baselines: Dict[str, List[float]] = defaultdict(list)
        self.pattern_history: Dict[str, List[MovementPattern]] = defaultdict(list)
        
        logger.info("BehaviorAnalyzer initialized")
    
    def _calculate_velocity_profile(self, trajectory: Trajectory) -> List[float]:
        """Calculate velocity profile from trajectory."""
        if len(trajectory.points) < 2:
            return []
        
        velocities = []
        for i in range(1, len(trajectory.points)):
            p1, p2 = trajectory.points[i-1], trajectory.points[i]
            
            # Calculate distance
            distance = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            
            # Calculate time difference
            if p1.timestamp and p2.timestamp:
                time_diff = (p2.timestamp - p1.timestamp).total_seconds()
                if time_diff > 0:
                    velocity = distance / time_diff  # pixels per second
                    velocities.append(velocity)
        
        return velocities
    
    def _calculate_direction_changes(self, trajectory: Trajectory) -> int:
        """Calculate number of significant direction changes."""
        if len(trajectory.points) < 3:
            return 0
        
        direction_changes = 0
        prev_angle = None
        
        for i in range(1, len(trajectory.points) - 1):
            p1, p2, p3 = trajectory.points[i-1], trajectory.points[i], trajectory.points[i+1]
            
            # Calculate vectors
            v1 = (p2.x - p1.x, p2.y - p1.y)
            v2 = (p3.x - p2.x, p3.y - p2.y)
            
            # Calculate angle between vectors
            if v1 != (0, 0) and v2 != (0, 0):
                angle1 = math.atan2(v1[1], v1[0])
                angle2 = math.atan2(v2[1], v2[0])
                
                angle_diff = abs(angle2 - angle1)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                
                # Consider significant if angle change > 45 degrees
                if angle_diff > math.pi / 4:
                    direction_changes += 1
        
        return direction_changes
    
    def _count_stops(self, trajectory: Trajectory) -> int:
        """Count number of stops in trajectory."""
        if len(trajectory.points) < 2:
            return 0
        
        stops = 0
        in_stop = False
        
        for i in range(1, len(trajectory.points)):
            p1, p2 = trajectory.points[i-1], trajectory.points[i]
            distance = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            
            if distance < self.movement_threshold:
                if not in_stop:
                    stops += 1
                    in_stop = True
            else:
                in_stop = False
        
        return stops
    
    def _calculate_area_coverage(self, trajectory: Trajectory) -> float:
        """Calculate area coverage of trajectory."""
        if len(trajectory.points) < 2:
            return 0.0
        
        x_coords = [p.x for p in trajectory.points]
        y_coords = [p.y for p in trajectory.points]
        
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        return width * height
    
    def _extract_movement_pattern(self, trajectory: Trajectory) -> MovementPattern:
        """Extract movement pattern from trajectory."""
        velocity_profile = self._calculate_velocity_profile(trajectory)
        direction_changes = self._calculate_direction_changes(trajectory)
        stops_count = self._count_stops(trajectory)
        total_distance = trajectory.get_distance_traveled()
        duration = trajectory.get_duration_seconds() or 0.0
        area_coverage = self._calculate_area_coverage(trajectory)
        
        return MovementPattern(
            velocity_profile=velocity_profile,
            direction_changes=direction_changes,
            stops_count=stops_count,
            total_distance=total_distance,
            duration_seconds=duration,
            area_coverage=area_coverage
        )
    
    def _point_in_bounds(self, point: Point, bounds: Tuple[float, float, float, float]) -> bool:
        """Check if point is within bounds."""
        x_min, y_min, x_max, y_max = bounds
        return x_min <= point.x <= x_max and y_min <= point.y <= y_max
    
    def analyze_loitering(self, trajectory: Trajectory, area_bounds: Tuple[float, float, float, float]) -> Dict[str, Any]:
        """Analyze trajectory for loitering behavior."""
        if not trajectory.points:
            return {
                "is_loitering": False,
                "loitering_score": 0.0,
                "time_in_area": 0.0,
                "movement_ratio": 0.0,
                "stationary_periods": [],
                "risk_level": "low"
            }
        
        # Calculate time spent in area
        points_in_area = [p for p in trajectory.points if self._point_in_bounds(p, area_bounds)]
        time_in_area = 0.0
        
        if len(points_in_area) > 1 and points_in_area[0].timestamp and points_in_area[-1].timestamp:
            time_in_area = (points_in_area[-1].timestamp - points_in_area[0].timestamp).total_seconds()
        
        # Calculate movement ratio (distance traveled / time spent)
        total_distance = trajectory.get_distance_traveled()
        movement_ratio = total_distance / time_in_area if time_in_area > 0 else 0.0
        
        # Detect stationary periods
        stationary_periods = []
        current_stationary_start = None
        
        for i, point in enumerate(trajectory.points):
            if not self._point_in_bounds(point, area_bounds):
                continue
            
            # Check if point is stationary
            is_stationary = True
            if i > 0:
                prev_point = trajectory.points[i-1]
                distance = math.sqrt((point.x - prev_point.x)**2 + (point.y - prev_point.y)**2)
                is_stationary = distance < self.movement_threshold
            
            if is_stationary:
                if current_stationary_start is None:
                    current_stationary_start = point.timestamp
            else:
                if current_stationary_start is not None:
                    stationary_periods.append((current_stationary_start, point.timestamp))
                    current_stationary_start = None
        
        # Close final stationary period if needed
        if current_stationary_start is not None and trajectory.points:
            stationary_periods.append((current_stationary_start, trajectory.points[-1].timestamp))
        
        # Calculate loitering score
        loitering_score = 0.0
        if time_in_area > 0:
            # Factors: time in area, low movement ratio, number of stationary periods
            time_factor = min(time_in_area / self.loitering_threshold, 1.0)
            movement_factor = max(0, 1.0 - movement_ratio / 50.0)  # Normalize movement
            stationary_factor = min(len(stationary_periods) / 3.0, 1.0)
            
            loitering_score = (time_factor * 0.5 + movement_factor * 0.3 + stationary_factor * 0.2)
        
        is_loitering = time_in_area >= self.loitering_threshold and loitering_score > 0.6
        
        # Determine risk level
        if loitering_score > 0.8:
            risk_level = "high"
        elif loitering_score > 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "is_loitering": is_loitering,
            "loitering_score": loitering_score,
            "time_in_area": time_in_area,
            "movement_ratio": movement_ratio,
            "stationary_periods": stationary_periods,
            "risk_level": risk_level
        }
    
    def detect_anomalous_movement(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Detect anomalous movement patterns."""
        if not trajectory.points:
            return {
                "is_anomalous": False,
                "anomaly_score": 0.0,
                "anomaly_types": [],
                "confidence": 0.0,
                "details": {}
            }
        
        pattern = self._extract_movement_pattern(trajectory)
        anomaly_types = []
        anomaly_scores = []
        details = {}
        
        # Check velocity anomalies
        if pattern.velocity_profile:
            avg_velocity = pattern.get_average_velocity()
            velocity_variance = pattern.get_velocity_variance()
            
            # Compare with historical baselines
            location_key = "global"  # Could be camera-specific
            historical_velocities = self.velocity_baselines[location_key]
            
            if len(historical_velocities) > 10:
                hist_mean = np.mean(historical_velocities)
                hist_std = np.std(historical_velocities)
                
                if hist_std > 0:
                    velocity_z_score = abs(avg_velocity - hist_mean) / hist_std
                    if velocity_z_score > self.velocity_anomaly_threshold:
                        anomaly_types.append("unusual_velocity")
                        anomaly_scores.append(min(velocity_z_score / 3.0, 1.0))
                        details["velocity_z_score"] = velocity_z_score
        
        # Check excessive direction changes
        if pattern.direction_changes > self.direction_change_threshold:
            anomaly_types.append("erratic_movement")
            direction_score = min(pattern.direction_changes / (self.direction_change_threshold * 2), 1.0)
            anomaly_scores.append(direction_score)
            details["direction_changes"] = pattern.direction_changes
        
        # Check unusual stop patterns
        if pattern.duration_seconds > 0:
            stop_ratio = pattern.stops_count / (pattern.duration_seconds / 60.0)  # stops per minute
            if stop_ratio > 2.0:  # More than 2 stops per minute
                anomaly_types.append("excessive_stops")
                stop_score = min(stop_ratio / 5.0, 1.0)
                anomaly_scores.append(stop_score)
                details["stop_ratio"] = stop_ratio
        
        # Check backtracking behavior
        if pattern.total_distance > 0 and pattern.area_coverage > 0:
            efficiency_ratio = pattern.area_coverage / (pattern.total_distance ** 2)
            if efficiency_ratio > 0.1:  # High area coverage relative to distance
                anomaly_types.append("backtracking")
                backtrack_score = min(efficiency_ratio * 10, 1.0)
                anomaly_scores.append(backtrack_score)
                details["efficiency_ratio"] = efficiency_ratio
        
        # Calculate overall anomaly score
        anomaly_score = max(anomaly_scores) if anomaly_scores else 0.0
        is_anomalous = anomaly_score > 0.6
        confidence = anomaly_score
        
        return {
            "is_anomalous": is_anomalous,
            "anomaly_score": anomaly_score,
            "anomaly_types": anomaly_types,
            "confidence": confidence,
            "details": details
        }
    
    def analyze_group_behavior(self, tracks: List[Track]) -> Dict[str, Any]:
        """Analyze behavior of multiple people moving together."""
        if len(tracks) < 2:
            return {
                "is_group": False,
                "group_size": len(tracks),
                "cohesion_score": 0.0,
                "coordination_score": 0.0,
                "group_type": "individual"
            }
        
        # Calculate spatial cohesion
        cohesion_scores = []
        coordination_scores = []
        
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                track1, track2 = tracks[i], tracks[j]
                
                if not track1.trajectory.points or not track2.trajectory.points:
                    continue
                
                # Calculate average distance between tracks
                distances = []
                min_points = min(len(track1.trajectory.points), len(track2.trajectory.points))
                
                for k in range(min_points):
                    p1 = track1.trajectory.points[k]
                    p2 = track2.trajectory.points[k]
                    distance = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                    distances.append(distance)
                
                if distances:
                    avg_distance = sum(distances) / len(distances)
                    distance_variance = sum((d - avg_distance)**2 for d in distances) / len(distances)
                    
                    # Cohesion: closer average distance = higher cohesion
                    cohesion = max(0, 1.0 - avg_distance / 200.0)  # Normalize by 200 pixels
                    cohesion_scores.append(cohesion)
                    
                    # Coordination: lower variance = higher coordination
                    coordination = max(0, 1.0 - distance_variance / 10000.0)  # Normalize
                    coordination_scores.append(coordination)
        
        avg_cohesion = sum(cohesion_scores) / len(cohesion_scores) if cohesion_scores else 0.0
        avg_coordination = sum(coordination_scores) / len(coordination_scores) if coordination_scores else 0.0
        
        # Determine if it's a group
        is_group = avg_cohesion > 0.5 and avg_coordination > 0.3
        
        # Classify group type
        group_type = "individual"
        if is_group:
            if len(tracks) <= 3:
                group_type = "small_group"
            elif len(tracks) <= 8:
                group_type = "medium_group"
            else:
                group_type = "large_group"
        
        return {
            "is_group": is_group,
            "group_size": len(tracks),
            "cohesion_score": avg_cohesion,
            "coordination_score": avg_coordination,
            "group_type": group_type
        }
    
    def calculate_risk_score(self, track: Track, context: Dict[str, Any]) -> float:
        """Calculate risk score based on behavior analysis."""
        if not track.trajectory.points:
            return 0.0
        
        risk_factors = []
        
        # Analyze loitering if area bounds provided
        if "area_bounds" in context:
            loitering_analysis = self.analyze_loitering(track.trajectory, context["area_bounds"])
            risk_factors.append(loitering_analysis["loitering_score"] * 0.4)
        
        # Analyze movement anomalies
        anomaly_analysis = self.detect_anomalous_movement(track.trajectory)
        risk_factors.append(anomaly_analysis["anomaly_score"] * 0.3)
        
        # Time-based factors
        current_time = datetime.now()
        if "high_risk_hours" in context:
            high_risk_hours = context["high_risk_hours"]
            current_hour = current_time.hour
            if current_hour in high_risk_hours:
                risk_factors.append(0.2)
        
        # Location-based factors
        if "high_risk_zones" in context and "current_zone" in context:
            if context["current_zone"] in context["high_risk_zones"]:
                risk_factors.append(0.3)
        
        # Duration factor
        duration = track.get_duration_seconds() or 0.0
        if duration > 600:  # More than 10 minutes
            duration_factor = min(duration / 1800.0, 0.3)  # Cap at 30 minutes
            risk_factors.append(duration_factor)
        
        # Calculate weighted risk score
        risk_score = sum(risk_factors) if risk_factors else 0.0
        return min(risk_score, 1.0)  # Cap at 1.0
    
    def get_historical_patterns(self, location: str, time_window_hours: int) -> Dict[str, Any]:
        """Get historical movement patterns for a location."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # Filter patterns by time window
        recent_patterns = []
        if location in self.pattern_history:
            # This is simplified - in practice, patterns would have timestamps
            recent_patterns = self.pattern_history[location][-100:]  # Last 100 patterns
        
        if not recent_patterns:
            return {
                "pattern_count": 0,
                "avg_velocity": 0.0,
                "avg_direction_changes": 0.0,
                "common_anomalies": []
            }
        
        # Calculate statistics
        velocities = []
        direction_changes = []
        
        for pattern in recent_patterns:
            if pattern.velocity_profile:
                velocities.extend(pattern.velocity_profile)
            direction_changes.append(pattern.direction_changes)
        
        avg_velocity = sum(velocities) / len(velocities) if velocities else 0.0
        avg_direction_changes = sum(direction_changes) / len(direction_changes) if direction_changes else 0.0
        
        return {
            "pattern_count": len(recent_patterns),
            "avg_velocity": avg_velocity,
            "avg_direction_changes": avg_direction_changes,
            "common_anomalies": []  # Could analyze common anomaly types
        }
    
    def update_baseline_behavior(self, tracks: List[Track], location: str) -> None:
        """Update baseline behavior patterns for a location."""
        for track in tracks:
            if not track.trajectory.points:
                continue
            
            pattern = self._extract_movement_pattern(track.trajectory)
            
            # Update pattern history
            self.pattern_history[location].append(pattern)
            
            # Keep only recent patterns (last 1000)
            if len(self.pattern_history[location]) > 1000:
                self.pattern_history[location] = self.pattern_history[location][-1000:]
            
            # Update velocity baselines
            if pattern.velocity_profile:
                self.velocity_baselines[location].extend(pattern.velocity_profile)
                
                # Keep only recent velocities (last 10000)
                if len(self.velocity_baselines[location]) > 10000:
                    self.velocity_baselines[location] = self.velocity_baselines[location][-10000:]
        
        logger.debug(f"Updated baseline behavior for location {location} with {len(tracks)} tracks")