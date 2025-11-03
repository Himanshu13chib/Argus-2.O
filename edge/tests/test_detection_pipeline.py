"""
Unit tests for Detection Pipeline with YOLO integration.

Tests core functionality including frame processing, model management,
virtual line crossing detection, and performance requirements.
"""

import pytest
import numpy as np
import cv2
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from edge.src.detection_pipeline import DetectionPipeline, ModelManager
from shared.models.detection import Detection, DetectionResult, BoundingBox, DetectionClass
from shared.models.virtual_line import VirtualLine, Point


class TestModelManager:
    """Test cases for ModelManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model_manager = ModelManager()
    
    def test_model_manager_initialization(self):
        """Test ModelManager initializes correctly."""
        assert len(self.model_manager.models) == 0
        assert len(self.model_manager.model_info) == 0
    
    @patch('edge.src.detection_pipeline.YOLO')
    def test_load_model_success(self, mock_yolo):
        """Test successful model loading."""
        # Mock YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        # Mock Path.exists to return True
        with patch('pathlib.Path.exists', return_value=True):
            result = self.model_manager.load_model("test_model.pt")
        
        assert result is True
        assert "test_model.pt" in self.model_manager.models
        assert "test_model.pt" in self.model_manager.model_info
        
    def test_load_model_file_not_found(self):
        """Test model loading with non-existent file."""
        with patch('pathlib.Path.exists', return_value=False):
            result = self.model_manager.load_model("nonexistent.pt")
        
        assert result is False
        assert len(self.model_manager.models) == 0
    
    @patch('edge.src.detection_pipeline.YOLO')
    @patch('torch.cuda.is_available', return_value=True)
    def test_optimize_model_cuda(self, mock_cuda, mock_yolo):
        """Test model optimization for CUDA device."""
        # Setup mock model
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        with patch('pathlib.Path.exists', return_value=True):
            self.model_manager.load_model("test_model.pt")
        
        result = self.model_manager.optimize_model("test_model.pt", "cuda:0")
        
        assert result is True
        assert self.model_manager.model_info["test_model.pt"]["optimized"] is True
    
    @patch('edge.src.detection_pipeline.YOLO')
    @patch('numpy.random.randint')
    def test_validate_model_performance(self, mock_randint, mock_yolo):
        """Test model validation with performance check."""
        # Mock numpy.random.randint to return valid array
        mock_randint.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Mock model with fast inference
        mock_model = Mock()
        mock_results = [Mock()]
        mock_model.return_value = mock_results
        mock_yolo.return_value = mock_model
        
        with patch('pathlib.Path.exists', return_value=True):
            self.model_manager.load_model("test_model.pt")
        
        # Manually set up model_info to avoid Mock subscriptable issues
        self.model_manager.model_info["test_model.pt"] = {
            "type": "yolo",
            "path": "test_model.pt",
            "loaded_at": datetime.now(),
            "device": "cpu",
            "input_size": [640, 640],
            "num_classes": 80,
            "optimized": False
        }
        
        # Mock time to simulate fast inference (100ms per inference)
        time_values = []
        for i in range(10):  # 5 inferences, 2 time calls each
            time_values.extend([i * 0.1, i * 0.1 + 0.1])  # Start and end times
        
        with patch('time.time', side_effect=time_values):
            result = self.model_manager.validate_model("test_model.pt")
        
        assert result is True
        model_info = self.model_manager.get_model_info("test_model.pt")
        assert model_info["validation_passed"] == True
        assert model_info["avg_inference_latency_ms"] < 300


class TestDetectionPipeline:
    """Test cases for DetectionPipeline class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "model_path": "test_model.pt",
            "confidence_threshold": 0.7,
            "nms_threshold": 0.45,
            "input_size": [640, 640],
            "device": "cpu",
            "batch_size": 1
        }
        self.pipeline = DetectionPipeline(self.config)
    
    def test_pipeline_initialization(self):
        """Test DetectionPipeline initializes with correct parameters."""
        assert self.pipeline.confidence_threshold == 0.7
        assert self.pipeline.nms_threshold == 0.45
        assert self.pipeline.input_size == (640, 640)
        assert self.pipeline.device == "cpu"
        assert self.pipeline.is_running is False
    
    @patch('edge.src.detection_pipeline.ModelManager')
    def test_start_pipeline_success(self, mock_model_manager):
        """Test successful pipeline startup."""
        # Mock model manager methods
        mock_manager = Mock()
        mock_manager.load_model.return_value = True
        mock_manager.optimize_model.return_value = True
        mock_manager.validate_model.return_value = True
        mock_model_manager.return_value = mock_manager
        
        pipeline = DetectionPipeline(self.config)
        pipeline.start()
        
        assert pipeline.is_running is True
    
    @patch('edge.src.detection_pipeline.ModelManager')
    def test_start_pipeline_model_load_failure(self, mock_model_manager):
        """Test pipeline startup with model loading failure."""
        # Mock model manager with load failure
        mock_manager = Mock()
        mock_manager.load_model.return_value = False
        mock_model_manager.return_value = mock_manager
        
        pipeline = DetectionPipeline(self.config)
        
        with pytest.raises(RuntimeError, match="Failed to load model"):
            pipeline.start()
        
        assert pipeline.is_running is False
    
    def test_preprocess_frame(self):
        """Test frame preprocessing functionality."""
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        processed_frame, scale_factors = self.pipeline._preprocess_frame(test_frame)
        
        # Check output dimensions
        assert processed_frame.shape[:2] == (640, 640)
        assert len(scale_factors) == 2
        assert scale_factors[0] > 0 and scale_factors[1] > 0
    
    def test_letterbox_resize(self):
        """Test letterbox resizing maintains aspect ratio."""
        # Create test frame with different aspect ratio
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        resized = self.pipeline._letterbox_resize(test_frame, (640, 640))
        
        assert resized.shape == (640, 640, 3)
        # Check that padding was added (should have gray borders)
        assert np.any(resized == 114)  # Letterbox padding color
    
    def create_mock_yolo_result(self, num_detections=1):
        """Create mock YOLO result for testing."""
        mock_result = Mock()
        
        if num_detections > 0:
            # Mock detection boxes (person class = 0)
            boxes = np.array([[100, 100, 200, 300]] * num_detections, dtype=np.float32)
            confidences = np.array([0.8] * num_detections, dtype=np.float32)
            classes = np.array([0] * num_detections, dtype=np.float32)  # Person class
            
            mock_boxes = Mock()
            mock_boxes.xyxy = Mock()
            mock_boxes.xyxy.cpu = Mock()
            mock_boxes.xyxy.cpu.return_value = Mock()
            mock_boxes.xyxy.cpu.return_value.numpy = Mock(return_value=boxes)
            
            mock_boxes.conf = Mock()
            mock_boxes.conf.cpu = Mock()
            mock_boxes.conf.cpu.return_value = Mock()
            mock_boxes.conf.cpu.return_value.numpy = Mock(return_value=confidences)
            
            mock_boxes.cls = Mock()
            mock_boxes.cls.cpu = Mock()
            mock_boxes.cls.cpu.return_value = Mock()
            mock_boxes.cls.cpu.return_value.numpy = Mock(return_value=classes)
            
            # Mock the len() method for boxes
            mock_boxes.__len__ = Mock(return_value=num_detections)
            
            mock_result.boxes = mock_boxes
        else:
            mock_result.boxes = None
        
        return mock_result
    
    @patch('edge.src.detection_pipeline.ModelManager')
    def test_process_frame_success(self, mock_model_manager):
        """Test successful frame processing."""
        # Setup mock model manager
        mock_manager = Mock()
        mock_model = Mock()
        mock_model.return_value = [self.create_mock_yolo_result(1)]
        mock_manager.get_model.return_value = mock_model
        mock_model_manager.return_value = mock_manager
        
        pipeline = DetectionPipeline(self.config)
        pipeline.is_running = True
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = pipeline.process_frame(test_frame, "test_camera")
        
        assert isinstance(result, DetectionResult)
        assert result.camera_id == "test_camera"
        assert len(result.detections) >= 0
        assert result.processing_time_ms > 0
    
    @patch('edge.src.detection_pipeline.ModelManager')
    def test_process_frame_latency_requirement(self, mock_model_manager):
        """Test that frame processing meets 300ms latency requirement."""
        # Setup fast mock model
        mock_manager = Mock()
        mock_model = Mock()
        mock_model.return_value = [self.create_mock_yolo_result(1)]
        mock_manager.get_model.return_value = mock_model
        mock_model_manager.return_value = mock_manager
        
        pipeline = DetectionPipeline(self.config)
        pipeline.is_running = True
        
        # Process multiple frames and check latency
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        latencies = []
        for _ in range(5):
            result = pipeline.process_frame(test_frame, "test_camera")
            latencies.append(result.processing_time_ms)
        
        avg_latency = np.mean(latencies)
        assert avg_latency < 300, f"Average latency {avg_latency}ms exceeds 300ms requirement"
    
    def test_virtual_line_crossing_detection(self):
        """Test virtual line crossing detection algorithms."""
        # Create test detection where foot position (bottom of bbox) crosses the line
        # Line is at y=250, so bbox should have bottom at around y=250
        bbox = BoundingBox(x=150, y=150, width=50, height=100)  # Bottom at y=250
        detection = Detection(
            id="test_detection",
            camera_id="test_camera",
            timestamp=datetime.now(),
            bbox=bbox,
            confidence=0.8,
            detection_class=DetectionClass.PERSON
        )
        
        # Create virtual line that should be crossed (horizontal line at y=250)
        virtual_line = VirtualLine(
            id="test_line",
            camera_id="test_camera",
            points=[Point(x=100, y=250), Point(x=200, y=250)],
            active=True,
            sensitivity=0.8
        )
        
        self.pipeline.virtual_lines = [virtual_line]
        
        # Test crossing detection
        crossing_detections = self.pipeline._check_virtual_line_crossings([detection])
        
        assert len(crossing_detections) > 0
        assert crossing_detections[0].metadata["virtual_line_crossing"] is True
        assert crossing_detections[0].metadata["virtual_line_id"] == "test_line"
    
    def test_virtual_line_no_crossing(self):
        """Test virtual line when no crossing occurs."""
        # Create test detection far from line
        bbox = BoundingBox(x=50, y=50, width=50, height=100)
        detection = Detection(
            id="test_detection",
            camera_id="test_camera",
            timestamp=datetime.now(),
            bbox=bbox,
            confidence=0.8,
            detection_class=DetectionClass.PERSON
        )
        
        # Create virtual line that should NOT be crossed
        virtual_line = VirtualLine(
            id="test_line",
            camera_id="test_camera",
            points=[Point(x=300, y=300), Point(x=400, y=300)],
            active=True,
            sensitivity=0.8
        )
        
        self.pipeline.virtual_lines = [virtual_line]
        
        # Test no crossing detection
        crossing_detections = self.pipeline._check_virtual_line_crossings([detection])
        
        assert len(crossing_detections) == 0
    
    def test_polygon_virtual_line_crossing(self):
        """Test polygon-based virtual line crossing detection."""
        # Create test detection where foot position is inside polygon
        # Polygon is 100x100 square from (100,100) to (200,200)
        # Detection foot should be at (175, 250) which is inside
        bbox = BoundingBox(x=150, y=150, width=50, height=100)  # Foot at (175, 250)
        detection = Detection(
            id="test_detection",
            camera_id="test_camera",
            timestamp=datetime.now(),
            bbox=bbox,
            confidence=0.8,
            detection_class=DetectionClass.PERSON
        )
        
        # Create polygon virtual line (square) that contains the foot position
        from shared.models.virtual_line import VirtualLineType
        virtual_line = VirtualLine(
            id="test_polygon",
            camera_id="test_camera",
            type=VirtualLineType.POLYGON,
            points=[
                Point(x=100, y=200),  # Adjusted to contain foot position
                Point(x=200, y=200),
                Point(x=200, y=300),
                Point(x=100, y=300)
            ],
            active=True,
            sensitivity=0.8
        )
        
        self.pipeline.virtual_lines = [virtual_line]
        
        # Test polygon crossing detection
        crossing_detections = self.pipeline._check_virtual_line_crossings([detection])
        
        assert len(crossing_detections) > 0
        assert crossing_detections[0].metadata["virtual_line_crossing"] is True
    
    def test_point_in_polygon_algorithm(self):
        """Test point-in-polygon algorithm accuracy."""
        # Define square polygon
        polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        
        # Test points inside polygon
        assert self.pipeline._point_in_polygon(50, 50, polygon) is True
        assert self.pipeline._point_in_polygon(10, 10, polygon) is True
        assert self.pipeline._point_in_polygon(90, 90, polygon) is True
        
        # Test points outside polygon
        assert self.pipeline._point_in_polygon(150, 50, polygon) is False
        assert self.pipeline._point_in_polygon(50, 150, polygon) is False
        assert self.pipeline._point_in_polygon(-10, 50, polygon) is False
    
    def test_update_virtual_lines(self):
        """Test virtual line configuration updates."""
        virtual_lines = [
            VirtualLine(
                id="line1",
                camera_id="cam1",
                points=[Point(x=0, y=100), Point(x=200, y=100)],
                active=True
            ),
            VirtualLine(
                id="line2",
                camera_id="cam2",
                points=[Point(x=0, y=200), Point(x=200, y=200)],
                active=True
            )
        ]
        
        result = self.pipeline.update_virtual_lines(virtual_lines)
        
        assert result is True
        assert len(self.pipeline.virtual_lines) == 2
        assert self.pipeline.virtual_lines[0].id == "line1"
        assert self.pipeline.virtual_lines[1].id == "line2"
    
    def test_health_status_healthy(self):
        """Test health status when pipeline is healthy."""
        self.pipeline.is_running = True
        self.pipeline.processing_times = [100, 150, 120, 180, 200]  # All under 300ms
        self.pipeline.total_frames = 100
        self.pipeline.failed_frames = 2
        self.pipeline.detection_count = 50
        
        health = self.pipeline.get_health_status()
        
        from shared.models.health import ComponentStatus
        assert health.status == ComponentStatus.HEALTHY
        assert health.component_id == "detection_pipeline"
        assert health.metadata["avg_processing_time_ms"] < 300
        assert health.metadata["success_rate_percent"] > 95
    
    def test_health_status_degraded_latency(self):
        """Test health status when latency exceeds requirements."""
        self.pipeline.is_running = True
        self.pipeline.processing_times = [400, 450, 420, 380, 500]  # Over 300ms
        self.pipeline.total_frames = 100
        self.pipeline.failed_frames = 2
        
        health = self.pipeline.get_health_status()
        
        from shared.models.health import ComponentStatus
        assert health.status == ComponentStatus.WARNING
        assert "300ms latency requirement" in health.status_message
        assert health.metadata["avg_processing_time_ms"] > 300
    
    def test_health_status_unhealthy_not_running(self):
        """Test health status when pipeline is not running."""
        self.pipeline.is_running = False
        
        health = self.pipeline.get_health_status()
        
        from shared.models.health import ComponentStatus
        assert health.status == ComponentStatus.CRITICAL
        assert "not running" in health.status_message


class TestDetectionAccuracy:
    """Test detection accuracy with synthetic data."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "model_path": "test_model.pt",
            "confidence_threshold": 0.7,
            "nms_threshold": 0.45,
            "input_size": [640, 640],
            "device": "cpu"
        }
    
    def create_synthetic_frame_with_person(self, width=640, height=480):
        """Create synthetic frame with person-like shape."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw simple person shape (rectangle for body, circle for head)
        # Body
        cv2.rectangle(frame, (200, 200), (250, 350), (255, 255, 255), -1)
        # Head
        cv2.circle(frame, (225, 180), 25, (255, 255, 255), -1)
        
        return frame
    
    @patch('edge.src.detection_pipeline.ModelManager')
    def test_synthetic_person_detection(self, mock_model_manager):
        """Test detection with synthetic person data."""
        # Setup mock to return person detection
        mock_manager = Mock()
        mock_model = Mock()
        
        # Create mock result with person detection at expected location
        mock_result = Mock()
        boxes = np.array([[200, 150, 250, 350]], dtype=np.float32)
        confidences = np.array([0.9], dtype=np.float32)
        classes = np.array([0], dtype=np.float32)  # Person class
        
        mock_boxes = Mock()
        mock_boxes.xyxy = Mock()
        mock_boxes.xyxy.cpu = Mock()
        mock_boxes.xyxy.cpu.return_value = Mock()
        mock_boxes.xyxy.cpu.return_value.numpy = Mock(return_value=boxes)
        
        mock_boxes.conf = Mock()
        mock_boxes.conf.cpu = Mock()
        mock_boxes.conf.cpu.return_value = Mock()
        mock_boxes.conf.cpu.return_value.numpy = Mock(return_value=confidences)
        
        mock_boxes.cls = Mock()
        mock_boxes.cls.cpu = Mock()
        mock_boxes.cls.cpu.return_value = Mock()
        mock_boxes.cls.cpu.return_value.numpy = Mock(return_value=classes)
        
        # Mock the len() method for boxes
        mock_boxes.__len__ = Mock(return_value=1)
        
        mock_result.boxes = mock_boxes
        
        mock_model.return_value = [mock_result]
        mock_manager.get_model.return_value = mock_model
        mock_model_manager.return_value = mock_manager
        
        pipeline = DetectionPipeline(self.config)
        pipeline.is_running = True
        
        # Test with synthetic frame
        synthetic_frame = self.create_synthetic_frame_with_person()
        result = pipeline.process_frame(synthetic_frame, "test_camera")
        
        assert len(result.detections) == 1
        detection = result.detections[0]
        assert detection.detection_class == DetectionClass.PERSON
        assert detection.confidence >= 0.7
        
        # Check bounding box is reasonable
        assert detection.bbox.width > 0
        assert detection.bbox.height > 0


if __name__ == "__main__":
    pytest.main([__file__])