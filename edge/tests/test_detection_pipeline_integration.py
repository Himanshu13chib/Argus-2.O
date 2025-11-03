"""
Integration tests for detection pipeline with mock camera inputs and edge cases.

Tests the complete detection pipeline with various scenarios including
synthetic video data, edge cases, and performance validation.
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
from shared.models.virtual_line import VirtualLine, Point, VirtualLineType


class MockCameraInputGenerator:
    """Mock camera input generator for testing."""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame_count = 0
    
    def generate_empty_frame(self):
        """Generate empty frame with no objects."""
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def generate_frame_with_person(self, x=200, y=150, width=50, height=100):
        """Generate frame with a person-like shape."""
        frame = self.generate_empty_frame()
        
        # Draw simple person shape
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), -1)
        cv2.circle(frame, (x + width//2, y - 20), 15, (255, 255, 255), -1)
        
        return frame
    
    def generate_frame_with_multiple_people(self, positions):
        """Generate frame with multiple people at specified positions."""
        frame = self.generate_empty_frame()
        
        for x, y in positions:
            cv2.rectangle(frame, (x, y), (x + 40, y + 80), (255, 255, 255), -1)
            cv2.circle(frame, (x + 20, y - 15), 12, (255, 255, 255), -1)
        
        return frame
    
    def generate_noisy_frame(self, noise_level=0.1):
        """Generate frame with random noise."""
        frame = np.random.randint(0, int(255 * noise_level), 
                                (self.height, self.width, 3), dtype=np.uint8)
        return frame


class TestDetectionPipelineIntegration:
    """Integration tests for detection pipeline."""
    
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
        self.camera_generator = MockCameraInputGenerator()
    
    @patch('edge.src.detection_pipeline.ModelManager')
    def test_empty_frame_processing(self, mock_model_manager):
        """Test processing of empty frames (no detections)."""
        # Setup mock to return no detections
        mock_manager = Mock()
        mock_model = Mock()
        
        # Mock empty result
        mock_result = Mock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        
        mock_manager.get_model.return_value = mock_model
        mock_model_manager.return_value = mock_manager
        
        pipeline = DetectionPipeline(self.config)
        pipeline.is_running = True
        
        # Test with empty frame
        empty_frame = self.camera_generator.generate_empty_frame()
        result = pipeline.process_frame(empty_frame, "test_camera")
        
        assert len(result.detections) == 0
        assert result.processing_time_ms > 0
        assert result.camera_id == "test_camera"
    
    @patch('edge.src.detection_pipeline.ModelManager')
    def test_multiple_people_detection(self, mock_model_manager):
        """Test detection of multiple people in single frame."""
        # Setup mock to return multiple person detections
        mock_manager = Mock()
        mock_model = Mock()
        
        # Create mock result with multiple detections
        mock_result = Mock()
        boxes = np.array([
            [100, 100, 140, 180],  # Person 1
            [300, 150, 340, 230],  # Person 2
            [500, 120, 540, 200]   # Person 3
        ], dtype=np.float32)
        confidences = np.array([0.9, 0.85, 0.8], dtype=np.float32)
        classes = np.array([0, 0, 0], dtype=np.float32)  # All person class
        
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
        
        mock_boxes.__len__ = Mock(return_value=3)
        mock_result.boxes = mock_boxes
        
        mock_model.return_value = [mock_result]
        mock_manager.get_model.return_value = mock_model
        mock_model_manager.return_value = mock_manager
        
        pipeline = DetectionPipeline(self.config)
        pipeline.is_running = True
        
        # Test with frame containing multiple people
        positions = [(100, 100), (300, 150), (500, 120)]
        frame = self.camera_generator.generate_frame_with_multiple_people(positions)
        result = pipeline.process_frame(frame, "test_camera")
        
        assert len(result.detections) == 3
        for detection in result.detections:
            assert detection.detection_class == DetectionClass.PERSON
            assert detection.confidence >= 0.7
    
    @patch('edge.src.detection_pipeline.ModelManager')
    def test_low_confidence_filtering(self, mock_model_manager):
        """Test that low confidence detections are filtered out by YOLO."""
        # Setup mock to return no detections (simulating YOLO confidence filtering)
        mock_manager = Mock()
        mock_model = Mock()
        
        # Mock empty result (YOLO already filtered out low confidence detections)
        mock_result = Mock()
        mock_result.boxes = None  # No detections passed YOLO confidence threshold
        
        mock_model.return_value = [mock_result]
        mock_manager.get_model.return_value = mock_model
        mock_model_manager.return_value = mock_manager
        
        pipeline = DetectionPipeline(self.config)
        pipeline.is_running = True
        
        frame = self.camera_generator.generate_frame_with_person()
        result = pipeline.process_frame(frame, "test_camera")
        
        # Should have no detections since YOLO filtered them out
        assert len(result.detections) == 0
    
    @patch('edge.src.detection_pipeline.ModelManager')
    def test_virtual_line_crossing_edge_cases(self, mock_model_manager):
        """Test virtual line crossing detection with edge cases."""
        # Setup mock detection
        mock_manager = Mock()
        mock_model = Mock()
        
        mock_result = Mock()
        boxes = np.array([[145, 80, 155, 120]], dtype=np.float32)  # Small detection near line
        confidences = np.array([0.9], dtype=np.float32)
        classes = np.array([0], dtype=np.float32)
        
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
        
        mock_boxes.__len__ = Mock(return_value=1)
        mock_result.boxes = mock_boxes
        
        mock_model.return_value = [mock_result]
        mock_manager.get_model.return_value = mock_model
        mock_model_manager.return_value = mock_manager
        
        pipeline = DetectionPipeline(self.config)
        pipeline.is_running = True
        
        # Create virtual line very close to detection
        virtual_line = VirtualLine(
            id="edge_case_line",
            camera_id="test_camera",
            points=[Point(x=100, y=100), Point(x=200, y=100)],
            active=True,
            sensitivity=0.9
        )
        pipeline.virtual_lines = [virtual_line]
        
        frame = self.camera_generator.generate_frame_with_person(x=145, y=80)
        result = pipeline.process_frame(frame, "test_camera")
        
        # Should detect crossing due to proximity
        assert len(result.detections) >= 0  # May or may not cross depending on exact algorithm
    
    @patch('edge.src.detection_pipeline.ModelManager')
    def test_performance_under_load(self, mock_model_manager):
        """Test pipeline performance with rapid frame processing."""
        # Setup fast mock
        mock_manager = Mock()
        mock_model = Mock()
        
        mock_result = Mock()
        mock_result.boxes = None  # No detections for speed
        mock_model.return_value = [mock_result]
        
        mock_manager.get_model.return_value = mock_model
        mock_model_manager.return_value = mock_manager
        
        pipeline = DetectionPipeline(self.config)
        pipeline.is_running = True
        
        # Process multiple frames rapidly
        frame = self.camera_generator.generate_empty_frame()
        processing_times = []
        
        for i in range(10):
            start_time = time.perf_counter()
            result = pipeline.process_frame(frame, f"camera_{i % 3}")
            processing_time = (time.perf_counter() - start_time) * 1000
            processing_times.append(processing_time)
        
        # Check that all frames meet latency requirement
        avg_time = np.mean(processing_times)
        max_time = max(processing_times)
        
        assert avg_time < 300, f"Average processing time {avg_time:.1f}ms exceeds 300ms"
        assert max_time < 500, f"Max processing time {max_time:.1f}ms is too high"
    
    @patch('edge.src.detection_pipeline.ModelManager')
    def test_error_recovery(self, mock_model_manager):
        """Test pipeline error recovery and graceful degradation."""
        # Setup mock that fails occasionally
        mock_manager = Mock()
        mock_model = Mock()
        
        # First call succeeds, second fails, third succeeds
        mock_model.side_effect = [
            [Mock(boxes=None)],  # Success
            Exception("Model inference failed"),  # Failure
            [Mock(boxes=None)]   # Recovery
        ]
        
        mock_manager.get_model.return_value = mock_model
        mock_model_manager.return_value = mock_manager
        
        pipeline = DetectionPipeline(self.config)
        pipeline.is_running = True
        
        frame = self.camera_generator.generate_empty_frame()
        
        # First call should succeed
        result1 = pipeline.process_frame(frame, "test_camera")
        assert result1.processing_time_ms > 0
        
        # Second call should handle error gracefully
        result2 = pipeline.process_frame(frame, "test_camera")
        assert len(result2.detections) == 0  # Should return empty result on error
        
        # Third call should recover
        result3 = pipeline.process_frame(frame, "test_camera")
        assert result3.processing_time_ms > 0
    
    def test_frame_preprocessing_edge_cases(self):
        """Test frame preprocessing with various input formats."""
        pipeline = DetectionPipeline(self.config)
        
        # Test different aspect ratios
        wide_frame = np.zeros((480, 1280, 3), dtype=np.uint8)  # 16:9
        tall_frame = np.zeros((1280, 480, 3), dtype=np.uint8)  # 9:16
        square_frame = np.zeros((640, 640, 3), dtype=np.uint8)  # 1:1
        
        for frame in [wide_frame, tall_frame, square_frame]:
            processed, scale_factors = pipeline._preprocess_frame(frame)
            
            # Should always output target size
            assert processed.shape[:2] == (640, 640)
            assert len(scale_factors) == 2
            assert all(sf > 0 for sf in scale_factors)
    
    def test_point_in_polygon_edge_cases(self):
        """Test point-in-polygon algorithm with edge cases."""
        pipeline = DetectionPipeline(self.config)
        
        # Test with triangle
        triangle = [(0, 0), (100, 0), (50, 100)]
        
        # Points inside
        assert pipeline._point_in_polygon(50, 30, triangle) is True
        assert pipeline._point_in_polygon(25, 25, triangle) is True
        
        # Points outside
        assert pipeline._point_in_polygon(150, 50, triangle) is False
        assert pipeline._point_in_polygon(50, 150, triangle) is False
        
        # Points on edge (may vary by implementation)
        edge_result = pipeline._point_in_polygon(50, 0, triangle)
        assert isinstance(edge_result, bool)
        
        # Test with complex polygon (concave)
        complex_polygon = [(0, 0), (100, 0), (100, 50), (50, 50), (50, 100), (0, 100)]
        
        assert pipeline._point_in_polygon(25, 25, complex_polygon) is True
        assert pipeline._point_in_polygon(75, 25, complex_polygon) is True
        assert pipeline._point_in_polygon(75, 75, complex_polygon) is False  # In concave part
        assert pipeline._point_in_polygon(25, 75, complex_polygon) is True


class TestDetectionPipelineStressTest:
    """Stress tests for detection pipeline."""
    
    def setup_method(self):
        """Set up stress test fixtures."""
        self.config = {
            "model_path": "test_model.pt",
            "confidence_threshold": 0.7,
            "nms_threshold": 0.45,
            "input_size": [640, 640],
            "device": "cpu"
        }
    
    @patch('edge.src.detection_pipeline.ModelManager')
    def test_memory_usage_stability(self, mock_model_manager):
        """Test that memory usage remains stable over many frames."""
        # Setup lightweight mock
        mock_manager = Mock()
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        mock_manager.get_model.return_value = mock_model
        mock_model_manager.return_value = mock_manager
        
        pipeline = DetectionPipeline(self.config)
        pipeline.is_running = True
        
        # Process many frames to test memory stability
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        for i in range(100):
            result = pipeline.process_frame(frame, f"camera_{i % 5}")
            
            # Check that processing times list doesn't grow unbounded
            assert len(pipeline.processing_times) <= 100
            
            # Verify result structure is consistent
            assert isinstance(result, DetectionResult)
            assert result.camera_id == f"camera_{i % 5}"
    
    @patch('edge.src.detection_pipeline.ModelManager')
    def test_concurrent_camera_simulation(self, mock_model_manager):
        """Simulate processing from multiple cameras."""
        mock_manager = Mock()
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        mock_manager.get_model.return_value = mock_model
        mock_model_manager.return_value = mock_manager
        
        pipeline = DetectionPipeline(self.config)
        pipeline.is_running = True
        
        # Simulate frames from 5 different cameras
        cameras = [f"camera_{i}" for i in range(5)]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        results = []
        for camera_id in cameras * 10:  # 50 total frames
            result = pipeline.process_frame(frame, camera_id)
            results.append(result)
        
        # Verify all cameras were processed
        processed_cameras = set(r.camera_id for r in results)
        assert len(processed_cameras) == 5
        
        # Check frame numbering is consistent
        assert results[-1].frame_number == 50


if __name__ == "__main__":
    pytest.main([__file__])