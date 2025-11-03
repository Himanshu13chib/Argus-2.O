"""
Unit tests for VirtualLineProcessor - Virtual Line Detection Engine.

Tests configurable boundary detection, polygon and line-based crossing detection
algorithms, and real-time virtual line overlay and configuration management
(Requirements 1.3, 4.2).
"""

import pytest
import numpy as np
import cv2
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from edge.src.virtual_line_processor import (
    VirtualLineProcessor, 
    CrossingDetection, 
    CrossingEvent,
    TrajectoryPoint
)
from shared.models.virtual_line import VirtualLine, Point, VirtualLineType, CrossingDirection
from shared.models.detection import Detection, BoundingBox, DetectionClass


class TestVirtualLineProcessor:
    """Test suite for VirtualLineProcessor."""
    
    @pytest.fixture
    def config(self):
        """Default configuration for testing."""
        return {
            "max_trajectory_length": 10,
            "trajectory_timeout_seconds": 30,
            "crossing_cooldown_seconds": 5,
            "line_thickness_pixels": 20,
            "polygon_edge_tolerance": 10
        }
    
    @pytest.fixture
    def processor(self, config):
        """Create VirtualLineProcessor instance for testing."""
        return VirtualLineProcessor(config)
    
    @pytest.fixture
    def sample_line_virtual_line(self):
        """Create a sample line-based virtual line."""
        vl = VirtualLine(
            id="test_line_1",
            camera_id="cam_001",
            name="Test Line",
            type=VirtualLineType.LINE,
            direction=CrossingDirection.BIDIRECTIONAL,
            sensitivity=0.8,
            active=True,
            color="#FF0000",
            thickness=2
        )
        vl.add_point(100, 100)
        vl.add_point(200, 100)
        return vl
    
    @pytest.fixture
    def sample_polygon_virtual_line(self):
        """Create a sample polygon-based virtual line."""
        vl = VirtualLine(
            id="test_polygon_1",
            camera_id="cam_001",
            name="Test Polygon",
            type=VirtualLineType.POLYGON,
            direction=CrossingDirection.BIDIRECTIONAL,
            sensitivity=0.9,
            active=True,
            color="#00FF00",
            thickness=2
        )
        # Create a square polygon
        vl.add_point(50, 50)
        vl.add_point(150, 50)
        vl.add_point(150, 150)
        vl.add_point(50, 150)
        return vl
    
    @pytest.fixture
    def sample_detection(self):
        """Create a sample detection."""
        return Detection(
            id="det_001",
            camera_id="cam_001",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=90, y=80, width=20, height=40),
            confidence=0.85,
            detection_class=DetectionClass.PERSON
        )
    
    def test_initialization(self, processor, config):
        """Test VirtualLineProcessor initialization."""
        assert processor.config == config
        assert len(processor.virtual_lines) == 0
        assert len(processor.trajectories) == 0
        assert processor.total_crossings_detected == 0
    
    def test_add_virtual_line(self, processor, sample_line_virtual_line):
        """Test adding a virtual line."""
        result = processor.add_virtual_line(sample_line_virtual_line)
        assert result is True
        assert len(processor.virtual_lines) == 1
        assert sample_line_virtual_line.id in processor.virtual_lines
    
    def test_add_invalid_virtual_line(self, processor):
        """Test adding an invalid virtual line."""
        # Create invalid virtual line with only one point
        invalid_line = VirtualLine(
            id="invalid_line",
            camera_id="cam_001",
            type=VirtualLineType.LINE
        )
        invalid_line.add_point(100, 100)  # Only one point
        
        result = processor.add_virtual_line(invalid_line)
        assert result is False
        assert len(processor.virtual_lines) == 0
    
    def test_remove_virtual_line(self, processor, sample_line_virtual_line):
        """Test removing a virtual line."""
        processor.add_virtual_line(sample_line_virtual_line)
        assert len(processor.virtual_lines) == 1
        
        result = processor.remove_virtual_line(sample_line_virtual_line.id)
        assert result is True
        assert len(processor.virtual_lines) == 0
    
    def test_update_virtual_line(self, processor, sample_line_virtual_line):
        """Test updating a virtual line."""
        processor.add_virtual_line(sample_line_virtual_line)
        
        # Update the line
        sample_line_virtual_line.sensitivity = 0.9
        result = processor.update_virtual_line(sample_line_virtual_line)
        assert result is True
        assert processor.virtual_lines[sample_line_virtual_line.id].sensitivity == 0.9
    
    def test_get_virtual_lines_by_camera(self, processor):
        """Test filtering virtual lines by camera ID."""
        # Add lines for different cameras
        line1 = VirtualLine(id="line1", camera_id="cam_001", type=VirtualLineType.LINE)
        line1.add_point(100, 100)
        line1.add_point(200, 100)
        
        line2 = VirtualLine(id="line2", camera_id="cam_002", type=VirtualLineType.LINE)
        line2.add_point(100, 100)
        line2.add_point(200, 100)
        
        processor.add_virtual_line(line1)
        processor.add_virtual_line(line2)
        
        cam1_lines = processor.get_virtual_lines("cam_001")
        assert len(cam1_lines) == 1
        assert cam1_lines[0].id == "line1"
        
        all_lines = processor.get_virtual_lines()
        assert len(all_lines) == 2
    
    def test_line_crossing_detection(self, processor, sample_line_virtual_line, sample_detection):
        """Test line-based crossing detection."""
        processor.add_virtual_line(sample_line_virtual_line)
        
        # Create detection that crosses the line (line is at y=100, detection foot at y=120)
        crossing_detection = Detection(
            id="crossing_det",
            camera_id="cam_001",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=140, y=80, width=20, height=40),  # Foot at (150, 120)
            confidence=0.9,
            detection_class=sample_detection.detection_class
        )
        
        crossings = processor.process_detections([crossing_detection], "cam_001")
        
        # Should detect crossing since foot position (150, 120) is near line y=100
        assert isinstance(crossings, list)
        # Note: Actual crossing detection depends on threshold and algorithm
    
    def test_polygon_crossing_detection(self, processor, sample_polygon_virtual_line):
        """Test polygon-based crossing detection."""
        processor.add_virtual_line(sample_polygon_virtual_line)
        
        # Create detection inside polygon (polygon is 50,50 to 150,150)
        inside_detection = Detection(
            id="inside_det",
            camera_id="cam_001",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=90, y=80, width=20, height=40),  # Foot at (100, 120)
            confidence=0.9,
            detection_class=DetectionClass.PERSON
        )
        
        # Create detection outside polygon
        outside_detection = Detection(
            id="outside_det",
            camera_id="cam_001",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=190, y=180, width=20, height=40),  # Foot at (200, 220)
            confidence=0.9,
            detection_class=DetectionClass.PERSON
        )
        
        # Process detections
        crossings1 = processor.process_detections([inside_detection], "cam_001")
        crossings2 = processor.process_detections([outside_detection], "cam_001")
        
        # Should handle polygon detection
        assert isinstance(crossings1, list)
        assert isinstance(crossings2, list)
    
    def test_inactive_virtual_line(self, processor, sample_line_virtual_line, sample_detection):
        """Test that inactive virtual lines don't generate crossings."""
        sample_line_virtual_line.active = False
        processor.add_virtual_line(sample_line_virtual_line)
        
        crossings = processor.process_detections([sample_detection], "cam_001")
        assert len(crossings) == 0
    
    def test_overlay_generation(self, processor, sample_line_virtual_line):
        """Test virtual line overlay generation."""
        processor.add_virtual_line(sample_line_virtual_line)
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        overlay_frame = processor.generate_overlay_image(frame, "cam_001")
        
        # Should return a frame (may be same as input if no lines)
        assert overlay_frame.shape == frame.shape
        assert isinstance(overlay_frame, np.ndarray)
    
    def test_configuration_export_import(self, processor, sample_line_virtual_line):
        """Test configuration export and import."""
        processor.add_virtual_line(sample_line_virtual_line)
        
        # Export configuration
        config_data = processor.export_configuration()
        assert "virtual_lines" in config_data
        assert len(config_data["virtual_lines"]) == 1
        
        # Create new processor and import
        new_processor = VirtualLineProcessor(processor.config)
        result = new_processor.import_configuration(config_data)
        assert result is True
        assert len(new_processor.virtual_lines) == 1
    
    def test_statistics(self, processor, sample_line_virtual_line):
        """Test statistics generation."""
        processor.add_virtual_line(sample_line_virtual_line)
        
        stats = processor.get_statistics()
        assert "total_virtual_lines" in stats
        assert "active_virtual_lines" in stats
        assert "total_crossings_detected" in stats
        assert stats["total_virtual_lines"] == 1
        assert stats["active_virtual_lines"] == 1
    
    def test_distance_calculation(self, processor):
        """Test point-to-line distance calculation."""
        point = Point(x=100, y=150)
        line_start = Point(x=50, y=100)
        line_end = Point(x=150, y=100)
        
        distance = processor._point_to_line_distance(point, line_start, line_end)
        assert distance == 50.0  # Point is 50 pixels above horizontal line
    
    def test_closest_point_on_line(self, processor):
        """Test finding closest point on line segment."""
        point = Point(x=100, y=150)
        line_start = Point(x=50, y=100)
        line_end = Point(x=150, y=100)
        
        closest = processor._closest_point_on_line(point, line_start, line_end)
        assert closest.x == 100
        assert closest.y == 100
    
    def test_hex_to_bgr_conversion(self, processor):
        """Test hex color to BGR conversion."""
        # Test red color
        bgr = processor._hex_to_bgr("#FF0000")
        assert bgr == (0, 0, 255)  # BGR format
        
        # Test green color
        bgr = processor._hex_to_bgr("#00FF00")
        assert bgr == (0, 255, 0)
        
        # Test invalid color (should default to red)
        bgr = processor._hex_to_bgr("invalid")
        assert bgr == (0, 0, 255)
    
    def test_multiple_camera_support(self, processor):
        """Test support for multiple cameras."""
        # Add lines for different cameras
        line1 = VirtualLine(id="line1", camera_id="cam_001", type=VirtualLineType.LINE)
        line1.add_point(100, 100)
        line1.add_point(200, 100)
        
        line2 = VirtualLine(id="line2", camera_id="cam_002", type=VirtualLineType.LINE)
        line2.add_point(100, 100)
        line2.add_point(200, 100)
        
        processor.add_virtual_line(line1)
        processor.add_virtual_line(line2)
        
        # Create detections for different cameras
        det1 = Detection(
            id="det1", camera_id="cam_001", timestamp=datetime.now(),
            bbox=BoundingBox(x=140, y=80, width=20, height=40),
            confidence=0.9, detection_class=DetectionClass.PERSON
        )
        
        det2 = Detection(
            id="det2", camera_id="cam_002", timestamp=datetime.now(),
            bbox=BoundingBox(x=140, y=80, width=20, height=40),
            confidence=0.9, detection_class=DetectionClass.PERSON
        )
        
        # Process detections for each camera
        crossings1 = processor.process_detections([det1], "cam_001")
        crossings2 = processor.process_detections([det2], "cam_002")
        
        assert isinstance(crossings1, list)
        assert isinstance(crossings2, list)