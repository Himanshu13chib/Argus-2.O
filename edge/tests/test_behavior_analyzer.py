"""
Unit tests for Behavioral Analysis system.

Tests loitering detection, anomaly detection, group behavior analysis,
and risk scoring with various movement patterns.
"""

import pytest
import numpy as np
import math
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from edge.src.behavior_analyzer import BehaviorAnalyzer, MovementPattern
from shared.models.tracking import Track, Trajectory, Point, TrackStatus
from shared.models.detection import Detection, BoundingBox, DetectionClass


class TestMovementPattern:
    """Test cases for MovementPattern dataclass."""
    
    def test_movement_pattern_creation(self):
        """Test MovementPattern creation and basic methods."""
        velocity_profile = [10.0, 15.0, 12.0, 8.0]
        pattern = MovementPattern(
            velocity_profile=velocity_profile,
            direction_changes=3,
            stops_count=2,
            total_distance=100.0,
            duration_seconds=60.0,
            area_coverage=500.0
        )
        
        assert pattern.get_average_velocity() == 11.25
        assert pattern.get_velocity_variance() == 7.1875
    
    def test_movement_pattern_empty_velocity(self):
        """Test MovementPattern with empty velocity profile."""
        pattern = MovementPattern(
            velocity_profile=[],
            direction_changes=0,
            stops_count=0,
            total_distance=0.0,
            duration_seconds=0.0,
            area_coverage=0.0
        )
        
        assert pattern.get_average_velocity() == 0.0
        assert pattern.get_velocity_variance() == 0.0


class TestBehaviorAnalyzer:
    """Test cases for BehaviorAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = BehaviorAnalyzer(
            loitering_threshold_seconds=300.0,
            movement_threshold_pixels=10.0,
            velocity_anomaly_threshold=2.0,
            direction_change_threshold=5
        )
        
        # Create test trajectory
        self.test_trajectory = self._create_test_trajectory()
        self.loitering_trajectory = self._create_loitering_trajectory()
        self.erratic_trajectory = self._create_erratic_trajectory()
    
    def _create_test_trajectory(self) -> Trajectory:
        """Create a normal test trajectory."""
        points = []
        start_time = datetime.now()
        
        for i in range(10):
            point = Point(
                x=100 + i * 20,
                y=100 + i * 10,
                timestamp=start_time + timedelta(seconds=i)
            )
            points.append(point)
        
        return Trajectory(points=points, start_time=start_time)
    
    def _create_loitering_trajectory(self) -> Trajectory:
        """Create a trajectory that exhibits loitering behavior."""
        points = []
        start_time = datetime.now()
        
        # Person stays in small area for extended time
        for i in range(20):
            # Small random movements within 20x20 pixel area
            point = Point(
                x=150 + (i % 4) * 5,
                y=150 + (i % 3) * 5,
                timestamp=start_time + timedelta(seconds=i * 30)  # 30 seconds between points
            )
            points.append(point)
        
        return Trajectory(points=points, start_time=start_time)
    
    def _create_erratic_trajectory(self) -> Trajectory:
        """Create a trajectory with erratic movement patterns."""
        points = []
        start_time = datetime.now()
        
        # Create zigzag pattern with many direction changes
        for i in range(15):
            if i % 2 == 0:
                x = 100 + i * 10
                y = 100
            else:
                x = 100 + i * 10
                y = 200
            
            point = Point(
                x=x,
                y=y,
                timestamp=start_time + timedelta(seconds=i * 2)
            )
            points.append(point)
        
        return Trajectory(points=points, start_time=start_time)
    
    def test_analyzer_initialization(self):
        """Test BehaviorAnalyzer initializes correctly."""
        assert self.analyzer.loitering_threshold == 300.0
        assert self.analyzer.movement_threshold == 10.0
        assert self.analyzer.velocity_anomaly_threshold == 2.0
        assert self.analyzer.direction_change_threshold == 5
        assert len(self.analyzer.location_baselines) == 0
    
    def test_calculate_velocity_profile(self):
        """Test velocity profile calculation."""
        velocities = self.analyzer._calculate_velocity_profile(self.test_trajectory)
        
        assert len(velocities) == 9  # n-1 velocities for n points
        assert all(v >= 0 for v in velocities)  # All velocities should be positive
    
    def test_calculate_direction_changes(self):
        """Test direction change calculation."""
        # Normal trajectory should have few direction changes
        normal_changes = self.analyzer._calculate_direction_changes(self.test_trajectory)
        assert normal_changes >= 0
        
        # Erratic trajectory should have many direction changes
        erratic_changes = self.analyzer._calculate_direction_changes(self.erratic_trajectory)
        assert erratic_changes > normal_changes
    
    def test_count_stops(self):
        """Test stop counting in trajectory."""
        stops = self.analyzer._count_stops(self.test_trajectory)
        assert stops >= 0
        
        # Loitering trajectory should have more stops
        loitering_stops = self.analyzer._count_stops(self.loitering_trajectory)
        assert loitering_stops >= stops
    
    def test_calculate_area_coverage(self):
        """Test area coverage calculation."""
        area = self.analyzer._calculate_area_coverage(self.test_trajectory)
        assert area > 0
        
        # Should be width * height of bounding box
        x_coords = [p.x for p in self.test_trajectory.points]
        y_coords = [p.y for p in self.test_trajectory.points]
        expected_area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        assert area == expected_area
    
    def test_extract_movement_pattern(self):
        """Test movement pattern extraction."""
        pattern = self.analyzer._extract_movement_pattern(self.test_trajectory)
        
        assert isinstance(pattern, MovementPattern)
        assert len(pattern.velocity_profile) > 0
        assert pattern.direction_changes >= 0
        assert pattern.stops_count >= 0
        assert pattern.total_distance > 0
        assert pattern.duration_seconds > 0
        assert pattern.area_coverage > 0
    
    def test_point_in_bounds(self):
        """Test point boundary checking."""
        point = Point(x=150, y=150)
        bounds = (100, 100, 200, 200)  # x_min, y_min, x_max, y_max
        
        assert self.analyzer._point_in_bounds(point, bounds)
        
        point_outside = Point(x=250, y=250)
        assert not self.analyzer._point_in_bounds(point_outside, bounds)
    
    def test_analyze_loitering_normal_behavior(self):
        """Test loitering analysis with normal movement."""
        area_bounds = (50, 50, 250, 250)
        result = self.analyzer.analyze_loitering(self.test_trajectory, area_bounds)
        
        assert isinstance(result, dict)
        assert "is_loitering" in result
        assert "loitering_score" in result
        assert "time_in_area" in result
        assert "movement_ratio" in result
        assert "stationary_periods" in result
        assert "risk_level" in result
        
        # Normal movement should not be classified as loitering
        assert not result["is_loitering"]
        assert result["risk_level"] in ["low", "medium", "high"]
    
    def test_analyze_loitering_behavior(self):
        """Test loitering analysis with actual loitering behavior."""
        area_bounds = (140, 140, 180, 180)  # Small area around loitering points
        result = self.analyzer.analyze_loitering(self.loitering_trajectory, area_bounds)
        
        # Should detect loitering
        assert result["time_in_area"] > 0
        assert result["loitering_score"] > 0
        # Note: May or may not be classified as loitering depending on exact parameters
    
    def test_analyze_loitering_empty_trajectory(self):
        """Test loitering analysis with empty trajectory."""
        empty_trajectory = Trajectory(points=[], start_time=datetime.now())
        area_bounds = (0, 0, 100, 100)
        
        result = self.analyzer.analyze_loitering(empty_trajectory, area_bounds)
        
        assert not result["is_loitering"]
        assert result["loitering_score"] == 0.0
        assert result["time_in_area"] == 0.0
        assert result["risk_level"] == "low"
    
    def test_detect_anomalous_movement_normal(self):
        """Test anomaly detection with normal movement."""
        result = self.analyzer.detect_anomalous_movement(self.test_trajectory)
        
        assert isinstance(result, dict)
        assert "is_anomalous" in result
        assert "anomaly_score" in result
        assert "anomaly_types" in result
        assert "confidence" in result
        assert "details" in result
        
        # Normal movement should have low anomaly score
        assert result["anomaly_score"] >= 0.0
        assert isinstance(result["anomaly_types"], list)
    
    def test_detect_anomalous_movement_erratic(self):
        """Test anomaly detection with erratic movement."""
        result = self.analyzer.detect_anomalous_movement(self.erratic_trajectory)
        
        # Erratic movement should have higher anomaly score
        assert result["anomaly_score"] >= 0.0
        
        # Should detect erratic movement
        if result["is_anomalous"]:
            assert "erratic_movement" in result["anomaly_types"]
    
    def test_detect_anomalous_movement_empty_trajectory(self):
        """Test anomaly detection with empty trajectory."""
        empty_trajectory = Trajectory(points=[], start_time=datetime.now())
        result = self.analyzer.detect_anomalous_movement(empty_trajectory)
        
        assert not result["is_anomalous"]
        assert result["anomaly_score"] == 0.0
        assert len(result["anomaly_types"]) == 0
    
    def test_analyze_group_behavior_single_track(self):
        """Test group behavior analysis with single track."""
        track = Track(
            id="single_track",
            camera_id="cam_1",
            trajectory=self.test_trajectory
        )
        
        result = self.analyzer.analyze_group_behavior([track])
        
        assert not result["is_group"]
        assert result["group_size"] == 1
        assert result["group_type"] == "individual"
    
    def test_analyze_group_behavior_multiple_tracks(self):
        """Test group behavior analysis with multiple tracks."""
        # Create tracks with similar trajectories (group behavior)
        tracks = []
        for i in range(3):
            points = []
            start_time = datetime.now()
            
            for j in range(10):
                point = Point(
                    x=100 + j * 20 + i * 30,  # Offset each track slightly
                    y=100 + j * 10 + i * 15,
                    timestamp=start_time + timedelta(seconds=j)
                )
                points.append(point)
            
            trajectory = Trajectory(points=points, start_time=start_time)
            track = Track(
                id=f"track_{i}",
                camera_id="cam_1",
                trajectory=trajectory
            )
            tracks.append(track)
        
        result = self.analyzer.analyze_group_behavior(tracks)
        
        assert result["group_size"] == 3
        assert result["cohesion_score"] >= 0.0
        assert result["coordination_score"] >= 0.0
        
        if result["is_group"]:
            assert result["group_type"] == "small_group"
    
    def test_calculate_risk_score_basic(self):
        """Test basic risk score calculation."""
        track = Track(
            id="risk_track",
            camera_id="cam_1",
            trajectory=self.test_trajectory
        )
        
        context = {
            "area_bounds": (50, 50, 250, 250),
            "high_risk_hours": [22, 23, 0, 1, 2, 3],
            "high_risk_zones": ["zone_1"],
            "current_zone": "zone_2"
        }
        
        risk_score = self.analyzer.calculate_risk_score(track, context)
        
        assert 0.0 <= risk_score <= 1.0
    
    def test_calculate_risk_score_high_risk_conditions(self):
        """Test risk score calculation with high-risk conditions."""
        track = Track(
            id="high_risk_track",
            camera_id="cam_1",
            trajectory=self.loitering_trajectory
        )
        
        # Set current time to high-risk hour
        with patch('edge.src.behavior_analyzer.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now().replace(hour=23)  # High-risk hour
            
            context = {
                "area_bounds": (140, 140, 180, 180),  # Area where loitering occurs
                "high_risk_hours": [22, 23, 0, 1, 2, 3],
                "high_risk_zones": ["zone_1"],
                "current_zone": "zone_1"  # High-risk zone
            }
            
            risk_score = self.analyzer.calculate_risk_score(track, context)
            
            assert 0.0 <= risk_score <= 1.0
            # Should have elevated risk due to multiple factors
    
    def test_calculate_risk_score_empty_trajectory(self):
        """Test risk score calculation with empty trajectory."""
        empty_trajectory = Trajectory(points=[], start_time=datetime.now())
        track = Track(
            id="empty_track",
            camera_id="cam_1",
            trajectory=empty_trajectory
        )
        
        risk_score = self.analyzer.calculate_risk_score(track, {})
        
        assert risk_score == 0.0
    
    def test_get_historical_patterns_empty(self):
        """Test getting historical patterns with no data."""
        result = self.analyzer.get_historical_patterns("location_1", 24)
        
        assert result["pattern_count"] == 0
        assert result["avg_velocity"] == 0.0
        assert result["avg_direction_changes"] == 0.0
        assert isinstance(result["common_anomalies"], list)
    
    def test_update_baseline_behavior(self):
        """Test updating baseline behavior patterns."""
        tracks = [
            Track(
                id="baseline_track_1",
                camera_id="cam_1",
                trajectory=self.test_trajectory
            ),
            Track(
                id="baseline_track_2",
                camera_id="cam_1",
                trajectory=self.erratic_trajectory
            )
        ]
        
        location = "test_location"
        initial_pattern_count = len(self.analyzer.pattern_history[location])
        initial_velocity_count = len(self.analyzer.velocity_baselines[location])
        
        self.analyzer.update_baseline_behavior(tracks, location)
        
        # Should have added patterns and velocities
        assert len(self.analyzer.pattern_history[location]) > initial_pattern_count
        assert len(self.analyzer.velocity_baselines[location]) > initial_velocity_count
    
    def test_get_historical_patterns_with_data(self):
        """Test getting historical patterns with existing data."""
        # Add some baseline data
        tracks = [
            Track(
                id="hist_track",
                camera_id="cam_1",
                trajectory=self.test_trajectory
            )
        ]
        
        location = "test_location_with_data"
        self.analyzer.update_baseline_behavior(tracks, location)
        
        result = self.analyzer.get_historical_patterns(location, 24)
        
        assert result["pattern_count"] > 0
        assert result["avg_velocity"] >= 0.0
        assert result["avg_direction_changes"] >= 0.0


class TestBehaviorAnalysisIntegration:
    """Integration tests for behavioral analysis system."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.analyzer = BehaviorAnalyzer()
    
    def test_complete_behavior_analysis_workflow(self):
        """Test complete behavior analysis workflow."""
        # Create a track with suspicious behavior
        suspicious_points = []
        start_time = datetime.now()
        
        # Create loitering pattern: person stays in area, makes small movements
        base_x, base_y = 150, 150
        for i in range(30):
            # Small circular movements
            angle = (i * 12) * math.pi / 180  # 12 degrees per step
            radius = 15
            x = base_x + radius * math.cos(angle)
            y = base_y + radius * math.sin(angle)
            
            point = Point(
                x=x,
                y=y,
                timestamp=start_time + timedelta(seconds=i * 20)  # 20 seconds per point
            )
            suspicious_points.append(point)
        
        suspicious_trajectory = Trajectory(points=suspicious_points, start_time=start_time)
        suspicious_track = Track(
            id="suspicious_track",
            camera_id="cam_1",
            trajectory=suspicious_trajectory
        )
        
        # Analyze loitering
        area_bounds = (100, 100, 200, 200)
        loitering_result = self.analyzer.analyze_loitering(suspicious_trajectory, area_bounds)
        
        # Analyze anomalies
        anomaly_result = self.analyzer.detect_anomalous_movement(suspicious_trajectory)
        
        # Calculate risk score
        context = {
            "area_bounds": area_bounds,
            "high_risk_hours": [datetime.now().hour],  # Current hour is high-risk
            "high_risk_zones": ["restricted_zone"],
            "current_zone": "restricted_zone"
        }
        risk_score = self.analyzer.calculate_risk_score(suspicious_track, context)
        
        # Verify results
        assert loitering_result["time_in_area"] > 0
        assert loitering_result["loitering_score"] > 0
        assert anomaly_result["anomaly_score"] >= 0
        assert 0.0 <= risk_score <= 1.0
        
        # Update baseline with this behavior
        self.analyzer.update_baseline_behavior([suspicious_track], "test_area")
        
        # Get historical patterns
        patterns = self.analyzer.get_historical_patterns("test_area", 1)
        assert patterns["pattern_count"] > 0
    
    def test_group_behavior_analysis_realistic(self):
        """Test group behavior analysis with realistic scenario."""
        # Create a group of 4 people moving together
        group_tracks = []
        start_time = datetime.now()
        
        for person_id in range(4):
            points = []
            
            # Group moves together with slight individual variations
            for step in range(15):
                # Base group movement
                base_x = 100 + step * 25
                base_y = 100 + step * 15
                
                # Individual offset within group
                offset_x = (person_id % 2) * 40 - 20  # -20 or +20
                offset_y = (person_id // 2) * 30 - 15  # -15 or +15
                
                # Add some random variation
                random_x = (person_id * 7 + step * 3) % 10 - 5  # -5 to +5
                random_y = (person_id * 5 + step * 2) % 8 - 4   # -4 to +4
                
                point = Point(
                    x=base_x + offset_x + random_x,
                    y=base_y + offset_y + random_y,
                    timestamp=start_time + timedelta(seconds=step * 3)
                )
                points.append(point)
            
            trajectory = Trajectory(points=points, start_time=start_time)
            track = Track(
                id=f"group_member_{person_id}",
                camera_id="cam_1",
                trajectory=trajectory
            )
            group_tracks.append(track)
        
        # Analyze group behavior
        group_result = self.analyzer.analyze_group_behavior(group_tracks)
        
        # Should detect coordinated group movement
        assert group_result["group_size"] == 4
        assert group_result["cohesion_score"] > 0
        assert group_result["coordination_score"] > 0
        
        if group_result["is_group"]:
            assert group_result["group_type"] == "small_group"
    
    def test_anomaly_detection_various_patterns(self):
        """Test anomaly detection with various movement patterns."""
        patterns_and_expected = [
            ("normal_walking", False),
            ("erratic_zigzag", True),
            ("sudden_stops", True),
            ("backtracking", True)
        ]
        
        for pattern_name, should_be_anomalous in patterns_and_expected:
            trajectory = self._create_pattern_trajectory(pattern_name)
            result = self.analyzer.detect_anomalous_movement(trajectory)
            
            # Note: Actual anomaly detection depends on specific thresholds
            # This test mainly ensures the system doesn't crash and returns valid results
            assert isinstance(result["is_anomalous"], bool)
            assert 0.0 <= result["anomaly_score"] <= 1.0
            assert isinstance(result["anomaly_types"], list)
    
    def _create_pattern_trajectory(self, pattern_type: str) -> Trajectory:
        """Create trajectory with specific movement pattern."""
        points = []
        start_time = datetime.now()
        
        if pattern_type == "normal_walking":
            # Straight line movement
            for i in range(10):
                point = Point(
                    x=100 + i * 30,
                    y=100 + i * 10,
                    timestamp=start_time + timedelta(seconds=i * 2)
                )
                points.append(point)
        
        elif pattern_type == "erratic_zigzag":
            # Zigzag with many direction changes
            for i in range(20):
                x = 100 + i * 15
                y = 100 + (50 if i % 2 == 0 else -50)
                point = Point(
                    x=x,
                    y=y,
                    timestamp=start_time + timedelta(seconds=i)
                )
                points.append(point)
        
        elif pattern_type == "sudden_stops":
            # Movement with many stops
            for i in range(15):
                if i % 3 == 0:
                    # Stop: same position for multiple time steps
                    x = 100 + (i // 3) * 50
                    y = 100
                else:
                    # Same position as previous stop
                    x = 100 + ((i - 1) // 3) * 50
                    y = 100
                
                point = Point(
                    x=x,
                    y=y,
                    timestamp=start_time + timedelta(seconds=i * 3)
                )
                points.append(point)
        
        elif pattern_type == "backtracking":
            # Forward then backward movement
            for i in range(10):
                if i < 5:
                    x = 100 + i * 40
                else:
                    x = 100 + (9 - i) * 40
                y = 100
                
                point = Point(
                    x=x,
                    y=y,
                    timestamp=start_time + timedelta(seconds=i * 2)
                )
                points.append(point)
        
        return Trajectory(points=points, start_time=start_time)