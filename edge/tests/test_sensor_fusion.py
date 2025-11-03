"""
Unit tests for multi-modal sensor fusion system.

Tests sensor fusion functionality including visible/thermal camera integration,
automatic lighting condition detection, and radar data integration.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from edge.src.sensor_fusion import (
    SensorFusion, SensorType, LightingCondition, WeatherCondition,
    SensorData, EnvironmentalConditions, RadarData
)
from shared.models.detection import Detection, DetectionClass, BoundingBox


class TestSensorFusion:
    """Test cases for SensorFusion class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "enabled_sensors": ["visible", "thermal", "radar"],
            "iou_threshold": 0.5,
            "confidence_boost": 0.1,
            "distance_threshold": 50.0
        }
        self.fusion = SensorFusion(self.config)
    
    def test_sensor_fusion_initialization(self):
        """Test SensorFusion initializes correctly."""
        assert len(self.fusion.active_sensors) == 3
        assert SensorType.VISIBLE in self.fusion.active_sensors
        assert SensorType.THERMAL in self.fusion.active_sensors
        assert SensorType.RADAR in self.fusion.active_sensors
        
        # Check weights are normalized
        total_weight = sum(self.fusion.sensor_weights.values())
        assert abs(total_weight - 1.0) < 0.001
    
    def create_test_detection(self, x=100, y=100, width=50, height=100, confidence=0.8, detection_id="test"):
        """Create a test detection for testing."""
        return Detection(
            id=detection_id,
            camera_id="test_camera",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=x, y=y, width=width, height=height),
            confidence=confidence,
            detection_class=DetectionClass.PERSON,
            metadata={}
        )
    
    def test_fuse_detections_single_sensor(self):
        """Test fusion with single sensor input."""
        visible_detections = [self.create_test_detection()]
        thermal_detections = []
        
        result = self.fusion.fuse_detections(visible_detections, thermal_detections)
        
        assert len(result) == 1
        assert result[0].detection_class == DetectionClass.PERSON
        assert "sensor_type" in result[0].metadata
        assert result[0].metadata["sensor_type"] == "visible"
    
    def test_fuse_detections_overlapping(self):
        """Test fusion of overlapping detections from different sensors."""
        # Create overlapping detections (same location, different sensors)
        visible_detection = self.create_test_detection(x=100, y=100, confidence=0.7, detection_id="vis_1")
        thermal_detection = self.create_test_detection(x=105, y=105, confidence=0.8, detection_id="therm_1")
        
        result = self.fusion.fuse_detections([visible_detection], [thermal_detection])
        
        # Should fuse into single detection
        assert len(result) == 1
        fused = result[0]
        
        # Check fusion metadata
        assert "fusion_type" in fused.metadata
        assert fused.metadata["fusion_type"] == "multi_sensor"
        assert "fused_sensors" in fused.metadata
        assert len(fused.metadata["fused_sensors"]) == 2
        
        # Confidence should be boosted due to fusion
        assert fused.confidence > max(visible_detection.confidence, thermal_detection.confidence)
    
    def test_fuse_detections_non_overlapping(self):
        """Test fusion of non-overlapping detections."""
        # Create non-overlapping detections
        visible_detection = self.create_test_detection(x=100, y=100, detection_id="vis_1")
        thermal_detection = self.create_test_detection(x=300, y=300, detection_id="therm_1")
        
        result = self.fusion.fuse_detections([visible_detection], [thermal_detection])
        
        # Should keep both detections separate
        assert len(result) == 2
        
        # Check that both detections are preserved
        detection_ids = [d.metadata.get("sensor_type") for d in result]
        assert "visible" in detection_ids
        assert "thermal" in detection_ids
    
    def test_calculate_iou(self):
        """Test IoU calculation for bounding boxes."""
        bbox1 = BoundingBox(x=0, y=0, width=100, height=100)
        bbox2 = BoundingBox(x=50, y=50, width=100, height=100)
        
        iou = self.fusion._calculate_iou(bbox1, bbox2)
        
        # Expected IoU for 50% overlap
        expected_iou = 2500 / (10000 + 10000 - 2500)  # intersection / union
        assert abs(iou - expected_iou) < 0.001
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation for non-overlapping boxes."""
        bbox1 = BoundingBox(x=0, y=0, width=50, height=50)
        bbox2 = BoundingBox(x=100, y=100, width=50, height=50)
        
        iou = self.fusion._calculate_iou(bbox1, bbox2)
        
        assert iou == 0.0
    
    def test_adapt_to_conditions_bright_light(self):
        """Test adaptation to bright lighting conditions."""
        self.fusion.adapt_to_conditions(lighting=150.0, weather="clear", temperature=25.0)
        
        assert self.fusion.environmental_conditions is not None
        assert self.fusion.environmental_conditions.lighting == LightingCondition.BRIGHT
        assert self.fusion.environmental_conditions.weather == WeatherCondition.CLEAR
        
        # In bright conditions, visible sensor should have higher weight
        assert self.fusion.sensor_weights[SensorType.VISIBLE] > self.fusion.sensor_weights[SensorType.THERMAL]
    
    def test_adapt_to_conditions_complete_darkness(self):
        """Test adaptation to complete darkness."""
        self.fusion.adapt_to_conditions(lighting=0.0, weather="clear", temperature=25.0)
        
        assert self.fusion.environmental_conditions.lighting == LightingCondition.COMPLETE_DARKNESS
        
        # In complete darkness, thermal should have higher weight than visible
        assert self.fusion.sensor_weights[SensorType.THERMAL] > self.fusion.sensor_weights[SensorType.VISIBLE]
        # Visible should have minimal or zero weight
        assert self.fusion.sensor_weights[SensorType.VISIBLE] <= 0.1
    
    def test_adapt_to_conditions_dust_storm(self):
        """Test adaptation to dust storm conditions."""
        initial_visible_weight = self.fusion.sensor_weights[SensorType.VISIBLE]
        
        self.fusion.adapt_to_conditions(lighting=50.0, weather="dust_storm", temperature=35.0)
        
        assert self.fusion.environmental_conditions.weather == WeatherCondition.DUST_STORM
        
        # Visible weight should be reduced in dust storm
        assert self.fusion.sensor_weights[SensorType.VISIBLE] < initial_visible_weight
    
    def test_classify_lighting_condition(self):
        """Test lighting condition classification."""
        assert self.fusion._classify_lighting_condition(150.0) == LightingCondition.BRIGHT
        assert self.fusion._classify_lighting_condition(50.0) == LightingCondition.NORMAL
        assert self.fusion._classify_lighting_condition(5.0) == LightingCondition.LOW_LIGHT
        assert self.fusion._classify_lighting_condition(0.5) == LightingCondition.DARK
        assert self.fusion._classify_lighting_condition(0.0) == LightingCondition.COMPLETE_DARKNESS
    
    def test_calculate_sensor_quality_visible(self):
        """Test sensor quality calculation for visible camera."""
        # Set bright conditions
        self.fusion.environmental_conditions = EnvironmentalConditions(
            lighting=LightingCondition.BRIGHT,
            weather=WeatherCondition.CLEAR,
            temperature=25.0,
            ambient_light_lux=150.0,
            visibility_meters=10000.0,
            timestamp=datetime.now()
        )
        
        quality = self.fusion._calculate_sensor_quality(SensorType.VISIBLE)
        assert quality == 1.0
        
        # Set dark conditions
        self.fusion.environmental_conditions.lighting = LightingCondition.COMPLETE_DARKNESS
        quality = self.fusion._calculate_sensor_quality(SensorType.VISIBLE)
        assert quality < 0.5
    
    def test_calculate_sensor_quality_thermal(self):
        """Test sensor quality calculation for thermal camera."""
        # Set normal temperature
        self.fusion.environmental_conditions = EnvironmentalConditions(
            lighting=LightingCondition.COMPLETE_DARKNESS,
            weather=WeatherCondition.CLEAR,
            temperature=25.0,
            ambient_light_lux=0.0,
            visibility_meters=10000.0,
            timestamp=datetime.now()
        )
        
        quality = self.fusion._calculate_sensor_quality(SensorType.THERMAL)
        assert quality == 1.0
        
        # Set extreme temperature
        self.fusion.environmental_conditions.temperature = 45.0
        quality = self.fusion._calculate_sensor_quality(SensorType.THERMAL)
        assert quality < 1.0
    
    def test_enhance_with_radar(self):
        """Test enhancement of detections with radar data."""
        detection = self.create_test_detection(x=100, y=100)
        
        radar_data = {
            "targets": [
                {
                    "x": 125,  # Close to detection center (125, 150)
                    "y": 150,
                    "range": 50.0,
                    "velocity": 2.5,
                    "angle": 45.0
                }
            ]
        }
        
        enhanced = self.fusion._enhance_with_radar([detection], radar_data)
        
        assert len(enhanced) == 1
        enhanced_detection = enhanced[0]
        
        # Should have radar enhancement metadata
        assert enhanced_detection.metadata.get("radar_enhanced") is True
        assert "radar_range" in enhanced_detection.metadata
        assert "radar_velocity" in enhanced_detection.metadata
        
        # Confidence should be boosted
        assert enhanced_detection.confidence > detection.confidence
    
    def test_enhance_with_radar_no_targets(self):
        """Test radar enhancement when no targets are found."""
        detection = self.create_test_detection()
        radar_data = {"targets": []}
        
        enhanced = self.fusion._enhance_with_radar([detection], radar_data)
        
        assert len(enhanced) == 1
        # Should be unchanged
        assert enhanced[0].confidence == detection.confidence
        assert enhanced[0].metadata.get("radar_enhanced") is not True
    
    def test_fallback_detection_selection(self):
        """Test fallback detection selection when fusion fails."""
        visible_detections = [self.create_test_detection(detection_id="vis")]
        thermal_detections = [self.create_test_detection(detection_id="therm")]
        
        # Test with dark conditions (should prefer thermal)
        self.fusion.environmental_conditions = EnvironmentalConditions(
            lighting=LightingCondition.COMPLETE_DARKNESS,
            weather=WeatherCondition.CLEAR,
            temperature=25.0,
            ambient_light_lux=0.0,
            visibility_meters=10000.0,
            timestamp=datetime.now()
        )
        
        result = self.fusion._fallback_detection_selection(visible_detections, thermal_detections)
        assert len(result) == 1
        # Should prefer thermal in dark conditions
        
        # Test with bright conditions (should prefer visible)
        self.fusion.environmental_conditions.lighting = LightingCondition.BRIGHT
        result = self.fusion._fallback_detection_selection(visible_detections, thermal_detections)
        assert len(result) == 1
    
    def test_get_active_sensors(self):
        """Test getting list of active sensors."""
        active_sensors = self.fusion.get_active_sensors()
        
        assert "visible" in active_sensors
        assert "thermal" in active_sensors
        assert "radar" in active_sensors
        assert len(active_sensors) == 3
    
    def test_calibrate_sensors(self):
        """Test sensor calibration functionality."""
        calibration_data = {
            "iou_threshold": 0.6,
            "confidence_boost": 0.15,
            "distance_threshold": 75.0,
            "sensor_weights": {
                "visible": 0.5,
                "thermal": 0.3,
                "radar": 0.2
            },
            "lighting_thresholds": {
                "bright": 120.0,
                "normal": 15.0
            }
        }
        
        result = self.fusion.calibrate_sensors(calibration_data)
        
        assert result is True
        assert self.fusion.iou_threshold == 0.6
        assert self.fusion.confidence_boost == 0.15
        assert self.fusion.distance_threshold == 75.0
        assert self.fusion.lighting_thresholds["bright"] == 120.0
        assert self.fusion.lighting_thresholds["normal"] == 15.0
    
    def test_environmental_confidence_adjustment(self):
        """Test environmental confidence adjustment calculation."""
        # Test clear bright conditions (should boost confidence)
        self.fusion.environmental_conditions = EnvironmentalConditions(
            lighting=LightingCondition.BRIGHT,
            weather=WeatherCondition.CLEAR,
            temperature=25.0,
            ambient_light_lux=150.0,
            visibility_meters=10000.0,
            timestamp=datetime.now()
        )
        
        adjustment = self.fusion._calculate_environmental_confidence_adjustment()
        assert adjustment > 0
        
        # Test dust storm conditions (should reduce confidence)
        self.fusion.environmental_conditions.weather = WeatherCondition.DUST_STORM
        adjustment = self.fusion._calculate_environmental_confidence_adjustment()
        assert adjustment < 0
    
    def test_apply_environmental_adaptation(self):
        """Test application of environmental adaptations to detections."""
        detection = self.create_test_detection(confidence=0.7)
        
        # Set conditions that boost confidence
        self.fusion.environmental_conditions = EnvironmentalConditions(
            lighting=LightingCondition.BRIGHT,
            weather=WeatherCondition.CLEAR,
            temperature=25.0,
            ambient_light_lux=150.0,
            visibility_meters=10000.0,
            timestamp=datetime.now()
        )
        
        adapted = self.fusion._apply_environmental_adaptation([detection])
        
        assert len(adapted) == 1
        adapted_detection = adapted[0]
        
        # Should have environmental adaptation metadata
        assert adapted_detection.metadata.get("environmental_adaptation") is True
        assert "confidence_adjustment" in adapted_detection.metadata
        assert "lighting_condition" in adapted_detection.metadata
        assert "weather_condition" in adapted_detection.metadata


class TestSensorFusionIntegration:
    """Integration tests for sensor fusion system."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.config = {
            "enabled_sensors": ["visible", "thermal"],
            "iou_threshold": 0.5,
            "confidence_boost": 0.1
        }
        self.fusion = SensorFusion(self.config)
    
    def test_complete_fusion_workflow(self):
        """Test complete sensor fusion workflow from detection to result."""
        # Create test detections from different sensors
        visible_detections = [
            Detection(
                id="vis_1",
                camera_id="cam_1",
                timestamp=datetime.now(),
                bbox=BoundingBox(x=100, y=100, width=50, height=100),
                confidence=0.75,
                detection_class=DetectionClass.PERSON
            ),
            Detection(
                id="vis_2",
                camera_id="cam_1",
                timestamp=datetime.now(),
                bbox=BoundingBox(x=300, y=200, width=60, height=120),
                confidence=0.65,
                detection_class=DetectionClass.PERSON
            )
        ]
        
        thermal_detections = [
            Detection(
                id="therm_1",
                camera_id="cam_1",
                timestamp=datetime.now(),
                bbox=BoundingBox(x=105, y=105, width=45, height=95),  # Overlaps with vis_1
                confidence=0.85,
                detection_class=DetectionClass.PERSON
            )
        ]
        
        # Set environmental conditions
        self.fusion.adapt_to_conditions(lighting=5.0, weather="clear", temperature=20.0)
        
        # Perform fusion
        result = self.fusion.fuse_detections(visible_detections, thermal_detections)
        
        # Should have 2 detections (1 fused, 1 standalone)
        assert len(result) == 2
        
        # Check for fused detection
        fused_detections = [d for d in result if d.metadata.get("fusion_type") == "multi_sensor"]
        assert len(fused_detections) == 1
        
        fused = fused_detections[0]
        assert fused.confidence > 0.75  # Should be boosted
        assert len(fused.metadata["fused_sensors"]) == 2
    
    def test_automatic_mode_switching(self):
        """Test automatic sensor mode switching based on lighting."""
        detection = Detection(
            id="test",
            camera_id="cam_1",
            timestamp=datetime.now(),
            bbox=BoundingBox(x=100, y=100, width=50, height=100),
            confidence=0.7,
            detection_class=DetectionClass.PERSON
        )
        
        # Test bright conditions
        self.fusion.adapt_to_conditions(lighting=200.0, weather="clear", temperature=25.0)
        visible_weight_bright = self.fusion.sensor_weights.get(SensorType.VISIBLE, 0)
        
        # Test dark conditions
        self.fusion.adapt_to_conditions(lighting=0.0, weather="clear", temperature=25.0)
        visible_weight_dark = self.fusion.sensor_weights.get(SensorType.VISIBLE, 0)
        thermal_weight_dark = self.fusion.sensor_weights.get(SensorType.THERMAL, 0)
        
        # Visible weight should be much higher in bright conditions
        assert visible_weight_bright > visible_weight_dark
        # Thermal weight should be higher in dark conditions
        assert thermal_weight_dark > 0.3


if __name__ == "__main__":
    pytest.main([__file__])