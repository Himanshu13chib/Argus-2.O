"""
Multi-modal sensor fusion system for Project Argus.

Implements sensor fusion for visible, thermal, and radar sensors with automatic
lighting condition detection and mode switching for enhanced detection accuracy
across all environmental conditions including complete darkness.
"""

import logging
import time
import numpy as np
import cv2
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass

from shared.interfaces.detection import ISensorFusion
from shared.models.detection import Detection, DetectionClass, BoundingBox


logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Types of sensors supported by the fusion system."""
    VISIBLE = "visible"
    THERMAL = "thermal"
    INFRARED = "infrared"
    RADAR = "radar"


class LightingCondition(Enum):
    """Lighting conditions for automatic mode switching."""
    BRIGHT = "bright"          # > 100 lux
    NORMAL = "normal"          # 10-100 lux
    LOW_LIGHT = "low_light"    # 1-10 lux
    DARK = "dark"              # < 1 lux
    COMPLETE_DARKNESS = "complete_darkness"  # 0 lux


class WeatherCondition(Enum):
    """Weather conditions affecting sensor performance."""
    CLEAR = "clear"
    RAIN = "rain"
    DUST_STORM = "dust_storm"
    FOG = "fog"
    SNOW = "snow"
    EXTREME_HEAT = "extreme_heat"


@dataclass
class SensorData:
    """Data from a single sensor."""
    sensor_type: SensorType
    frame: np.ndarray
    detections: List[Detection]
    timestamp: datetime
    quality_score: float = 1.0
    metadata: Dict[str, Any] = None


@dataclass
class EnvironmentalConditions:
    """Current environmental conditions."""
    lighting: LightingCondition
    weather: WeatherCondition
    temperature: float  # Celsius
    ambient_light_lux: float
    visibility_meters: float
    timestamp: datetime


@dataclass
class RadarData:
    """Radar sensor data for enhanced detection."""
    range_data: np.ndarray  # Distance measurements
    velocity_data: np.ndarray  # Velocity measurements
    angle_data: np.ndarray  # Angle measurements
    targets: List[Dict[str, Any]]  # Detected targets
    timestamp: datetime
    quality: float = 1.0


class SensorFusion(ISensorFusion):
    """
    Multi-modal sensor fusion system for enhanced detection accuracy.
    
    Supports visible, thermal, infrared, and radar sensors with automatic
    environmental adaptation and mode switching (Requirements 2.1, 2.2, 2.3, 2.4, 2.7).
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_sensors: List[SensorType] = []
        self.sensor_weights: Dict[SensorType, float] = {}
        self.environmental_conditions: Optional[EnvironmentalConditions] = None
        
        # Fusion parameters
        self.iou_threshold = config.get("iou_threshold", 0.5)
        self.confidence_boost = config.get("confidence_boost", 0.1)
        self.distance_threshold = config.get("distance_threshold", 50.0)  # pixels
        
        # Lighting thresholds for automatic switching
        self.lighting_thresholds = {
            "bright": 100.0,      # > 100 lux
            "normal": 10.0,       # 10-100 lux
            "low_light": 1.0,     # 1-10 lux
            "dark": 0.1,          # 0.1-1 lux
            "complete_darkness": 0.0  # 0 lux
        }
        
        # Initialize sensor weights based on configuration
        self._initialize_sensor_weights()
        
        logger.info("SensorFusion initialized with multi-modal support")
        logger.info(f"Active sensors: {[s.value for s in self.active_sensors]}")
    
    def _initialize_sensor_weights(self) -> None:
        """Initialize sensor weights based on configuration."""
        # Default weights for different sensors
        default_weights = {
            SensorType.VISIBLE: 0.4,
            SensorType.THERMAL: 0.3,
            SensorType.INFRARED: 0.2,
            SensorType.RADAR: 0.1
        }
        
        # Get configured sensors
        enabled_sensors = self.config.get("enabled_sensors", ["visible", "thermal"])
        
        for sensor_name in enabled_sensors:
            try:
                sensor_type = SensorType(sensor_name)
                self.active_sensors.append(sensor_type)
                self.sensor_weights[sensor_type] = default_weights.get(sensor_type, 0.25)
            except ValueError:
                logger.warning(f"Unknown sensor type: {sensor_name}")
        
        # Normalize weights
        total_weight = sum(self.sensor_weights.values())
        if total_weight > 0:
            for sensor_type in self.sensor_weights:
                self.sensor_weights[sensor_type] /= total_weight
    
    def fuse_detections(self, visible: List[Detection], thermal: List[Detection], 
                       radar_data: Optional[Dict[str, Any]] = None) -> List[Detection]:
        """
        Fuse detections from multiple sensor modalities (Requirement 2.1, 2.2).
        
        Args:
            visible: Detections from visible light camera
            thermal: Detections from thermal camera
            radar_data: Optional radar data for enhanced detection
            
        Returns:
            List of fused detections with enhanced confidence scores
        """
        try:
            # Prepare sensor data
            sensor_data = []
            
            if visible and SensorType.VISIBLE in self.active_sensors:
                sensor_data.append(SensorData(
                    sensor_type=SensorType.VISIBLE,
                    frame=None,  # Frame not needed for fusion
                    detections=visible,
                    timestamp=datetime.now(),
                    quality_score=self._calculate_sensor_quality(SensorType.VISIBLE)
                ))
            
            if thermal and SensorType.THERMAL in self.active_sensors:
                sensor_data.append(SensorData(
                    sensor_type=SensorType.THERMAL,
                    frame=None,
                    detections=thermal,
                    timestamp=datetime.now(),
                    quality_score=self._calculate_sensor_quality(SensorType.THERMAL)
                ))
            
            # Perform multi-modal fusion
            fused_detections = self._perform_detection_fusion(sensor_data)
            
            # Enhance with radar data if available
            if radar_data and SensorType.RADAR in self.active_sensors:
                fused_detections = self._enhance_with_radar(fused_detections, radar_data)
            
            # Apply environmental adaptation
            if self.environmental_conditions:
                fused_detections = self._apply_environmental_adaptation(fused_detections)
            
            logger.debug(f"Fused {len(visible)} visible + {len(thermal)} thermal detections "
                        f"into {len(fused_detections)} final detections")
            
            return fused_detections
            
        except Exception as e:
            logger.error(f"Error in sensor fusion: {e}")
            # Fallback to best available sensor
            return self._fallback_detection_selection(visible, thermal)
    
    def _perform_detection_fusion(self, sensor_data: List[SensorData]) -> List[Detection]:
        """
        Perform the core detection fusion algorithm.
        
        Args:
            sensor_data: List of sensor data with detections
            
        Returns:
            List of fused detections
        """
        if not sensor_data:
            return []
        
        if len(sensor_data) == 1:
            # Even for single sensor, add sensor metadata
            data = sensor_data[0]
            enhanced_detections = []
            for detection in data.detections:
                enhanced_detection = Detection(
                    id=detection.id,
                    camera_id=detection.camera_id,
                    timestamp=detection.timestamp,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    detection_class=detection.detection_class,
                    features=detection.features,
                    image_crop=detection.image_crop,
                    metadata={
                        **(detection.metadata or {}),
                        "sensor_type": data.sensor_type.value,
                        "sensor_quality": data.quality_score,
                        "sensor_weight": self.sensor_weights.get(data.sensor_type, 0.25)
                    }
                )
                enhanced_detections.append(enhanced_detection)
            return enhanced_detections
        
        # Collect all detections with sensor information
        all_detections = []
        for data in sensor_data:
            for detection in data.detections:
                # Add sensor metadata to detection
                enhanced_detection = Detection(
                    id=detection.id,
                    camera_id=detection.camera_id,
                    timestamp=detection.timestamp,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    detection_class=detection.detection_class,
                    features=detection.features,
                    image_crop=detection.image_crop,
                    metadata={
                        **(detection.metadata or {}),
                        "sensor_type": data.sensor_type.value,
                        "sensor_quality": data.quality_score,
                        "sensor_weight": self.sensor_weights.get(data.sensor_type, 0.25)
                    }
                )
                all_detections.append(enhanced_detection)
        
        # Group overlapping detections from different sensors
        fused_detections = []
        used_detections = set()
        
        for i, detection1 in enumerate(all_detections):
            if i in used_detections:
                continue
            
            # Find overlapping detections from other sensors
            overlapping_group = [detection1]
            used_detections.add(i)
            
            for j, detection2 in enumerate(all_detections[i+1:], i+1):
                if j in used_detections:
                    continue
                
                # Check if detections overlap (same object from different sensors)
                if self._detections_overlap(detection1, detection2):
                    overlapping_group.append(detection2)
                    used_detections.add(j)
            
            # Fuse the overlapping detections
            if len(overlapping_group) > 1:
                fused_detection = self._fuse_detection_group(overlapping_group)
            else:
                fused_detection = overlapping_group[0]
            
            fused_detections.append(fused_detection)
        
        return fused_detections
    
    def _detections_overlap(self, det1: Detection, det2: Detection) -> bool:
        """Check if two detections overlap (represent the same object)."""
        # Calculate IoU (Intersection over Union)
        iou = self._calculate_iou(det1.bbox, det2.bbox)
        return iou > self.iou_threshold
    
    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        # Calculate intersection
        x1 = max(bbox1.x, bbox2.x)
        y1 = max(bbox1.y, bbox2.y)
        x2 = min(bbox1.x + bbox1.width, bbox2.x + bbox2.width)
        y2 = min(bbox1.y + bbox1.height, bbox2.y + bbox2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = bbox1.width * bbox1.height
        area2 = bbox2.width * bbox2.height
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def _fuse_detection_group(self, detections: List[Detection]) -> Detection:
        """
        Fuse a group of overlapping detections from different sensors.
        
        Args:
            detections: List of overlapping detections to fuse
            
        Returns:
            Single fused detection with enhanced confidence
        """
        if len(detections) == 1:
            return detections[0]
        
        # Calculate weighted average of bounding boxes
        total_weight = 0
        weighted_x = 0
        weighted_y = 0
        weighted_width = 0
        weighted_height = 0
        weighted_confidence = 0
        
        sensor_types = []
        all_metadata = {}
        
        for detection in detections:
            sensor_type_str = detection.metadata.get("sensor_type", "unknown")
            sensor_weight = detection.metadata.get("sensor_weight", 0.25)
            sensor_quality = detection.metadata.get("sensor_quality", 1.0)
            
            # Adjust weight by sensor quality
            effective_weight = sensor_weight * sensor_quality
            
            weighted_x += detection.bbox.x * effective_weight
            weighted_y += detection.bbox.y * effective_weight
            weighted_width += detection.bbox.width * effective_weight
            weighted_height += detection.bbox.height * effective_weight
            weighted_confidence += detection.confidence * effective_weight
            
            total_weight += effective_weight
            sensor_types.append(sensor_type_str)
            
            # Merge metadata
            if detection.metadata:
                all_metadata.update(detection.metadata)
        
        if total_weight == 0:
            total_weight = 1
        
        # Create fused bounding box
        fused_bbox = BoundingBox(
            x=weighted_x / total_weight,
            y=weighted_y / total_weight,
            width=weighted_width / total_weight,
            height=weighted_height / total_weight
        )
        
        # Enhanced confidence from multi-sensor fusion
        base_confidence = weighted_confidence / total_weight
        fusion_boost = min(0.2, len(detections) * self.confidence_boost)
        final_confidence = min(1.0, base_confidence + fusion_boost)
        
        # Use the detection with highest confidence as base
        base_detection = max(detections, key=lambda d: d.confidence)
        
        # Create fused detection
        fused_detection = Detection(
            id=f"fused_{base_detection.id}_{int(time.time() * 1000)}",
            camera_id=base_detection.camera_id,
            timestamp=datetime.now(),
            bbox=fused_bbox,
            confidence=final_confidence,
            detection_class=base_detection.detection_class,
            features=base_detection.features,
            image_crop=base_detection.image_crop,
            metadata={
                **all_metadata,
                "fusion_type": "multi_sensor",
                "fused_sensors": sensor_types,
                "fusion_count": len(detections),
                "original_confidences": [d.confidence for d in detections],
                "fusion_boost": fusion_boost,
                "fusion_timestamp": datetime.now().isoformat()
            }
        )
        
        return fused_detection
    
    def _enhance_with_radar(self, detections: List[Detection], radar_data: Dict[str, Any]) -> List[Detection]:
        """
        Enhance detections with radar data for improved accuracy.
        
        Args:
            detections: Visual/thermal detections
            radar_data: Radar sensor data
            
        Returns:
            Enhanced detections with radar information
        """
        try:
            enhanced_detections = []
            
            # Extract radar targets
            radar_targets = radar_data.get("targets", [])
            
            for detection in detections:
                enhanced_detection = detection
                
                # Find closest radar target
                closest_target = self._find_closest_radar_target(detection, radar_targets)
                
                if closest_target:
                    # Enhance detection with radar data
                    radar_confidence_boost = 0.1
                    enhanced_confidence = min(1.0, detection.confidence + radar_confidence_boost)
                    
                    enhanced_detection = Detection(
                        id=detection.id,
                        camera_id=detection.camera_id,
                        timestamp=detection.timestamp,
                        bbox=detection.bbox,
                        confidence=enhanced_confidence,
                        detection_class=detection.detection_class,
                        features=detection.features,
                        image_crop=detection.image_crop,
                        metadata={
                            **(detection.metadata or {}),
                            "radar_enhanced": True,
                            "radar_range": closest_target.get("range", 0),
                            "radar_velocity": closest_target.get("velocity", 0),
                            "radar_angle": closest_target.get("angle", 0),
                            "radar_confidence_boost": radar_confidence_boost
                        }
                    )
                
                enhanced_detections.append(enhanced_detection)
            
            return enhanced_detections
            
        except Exception as e:
            logger.error(f"Error enhancing with radar data: {e}")
            return detections
    
    def _find_closest_radar_target(self, detection: Detection, radar_targets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the closest radar target to a visual detection."""
        if not radar_targets:
            return None
        
        # Convert detection center to radar coordinates (simplified)
        detection_center_x = detection.bbox.x + detection.bbox.width / 2
        detection_center_y = detection.bbox.y + detection.bbox.height / 2
        
        closest_target = None
        min_distance = float('inf')
        
        for target in radar_targets:
            # Simplified coordinate mapping (would need proper calibration in production)
            target_x = target.get("x", 0)
            target_y = target.get("y", 0)
            
            distance = np.sqrt((detection_center_x - target_x) ** 2 + (detection_center_y - target_y) ** 2)
            
            if distance < min_distance and distance < self.distance_threshold:
                min_distance = distance
                closest_target = target
        
        return closest_target
    
    def _calculate_sensor_quality(self, sensor_type: SensorType) -> float:
        """Calculate current quality score for a sensor based on conditions."""
        if not self.environmental_conditions:
            return 1.0
        
        conditions = self.environmental_conditions
        
        if sensor_type == SensorType.VISIBLE:
            # Visible camera quality depends on lighting
            if conditions.lighting in [LightingCondition.BRIGHT, LightingCondition.NORMAL]:
                return 1.0
            elif conditions.lighting == LightingCondition.LOW_LIGHT:
                return 0.6
            else:  # DARK or COMPLETE_DARKNESS
                return 0.2
        
        elif sensor_type == SensorType.THERMAL:
            # Thermal camera works well in all lighting but affected by temperature
            base_quality = 1.0
            if conditions.temperature > 40:  # Very hot conditions
                base_quality *= 0.8
            elif conditions.temperature < -20:  # Very cold conditions
                base_quality *= 0.9
            return base_quality
        
        elif sensor_type == SensorType.INFRARED:
            # Infrared works well in darkness
            if conditions.lighting in [LightingCondition.DARK, LightingCondition.COMPLETE_DARKNESS]:
                return 1.0
            else:
                return 0.7
        
        elif sensor_type == SensorType.RADAR:
            # Radar not affected by lighting or weather (mostly)
            if conditions.weather == WeatherCondition.EXTREME_HEAT:
                return 0.9
            return 1.0
        
        return 1.0
    
    def _fallback_detection_selection(self, visible: List[Detection], thermal: List[Detection]) -> List[Detection]:
        """Select best available detections when fusion fails."""
        if not self.environmental_conditions:
            # Default to visible if no environmental data
            return visible if visible else thermal
        
        # Select based on lighting conditions
        if self.environmental_conditions.lighting in [LightingCondition.DARK, LightingCondition.COMPLETE_DARKNESS]:
            return thermal if thermal else visible
        else:
            return visible if visible else thermal
    
    def adapt_to_conditions(self, lighting: float, weather: str, temperature: float) -> None:
        """
        Adapt fusion parameters based on environmental conditions (Requirement 2.4, 2.6, 2.7).
        
        Args:
            lighting: Ambient light level in lux
            weather: Weather condition string
            temperature: Temperature in Celsius
        """
        try:
            # Determine lighting condition
            lighting_condition = self._classify_lighting_condition(lighting)
            
            # Parse weather condition
            try:
                weather_condition = WeatherCondition(weather.lower())
            except ValueError:
                weather_condition = WeatherCondition.CLEAR
                logger.warning(f"Unknown weather condition: {weather}, defaulting to clear")
            
            # Update environmental conditions
            self.environmental_conditions = EnvironmentalConditions(
                lighting=lighting_condition,
                weather=weather_condition,
                temperature=temperature,
                ambient_light_lux=lighting,
                visibility_meters=self._estimate_visibility(weather_condition),
                timestamp=datetime.now()
            )
            
            # Adapt sensor weights based on conditions (Requirement 2.4)
            self._adapt_sensor_weights()
            
            logger.info(f"Adapted to conditions: lighting={lighting_condition.value}, "
                       f"weather={weather_condition.value}, temp={temperature}°C")
            
        except Exception as e:
            logger.error(f"Error adapting to conditions: {e}")
    
    def _classify_lighting_condition(self, lux: float) -> LightingCondition:
        """Classify lighting condition based on lux value."""
        if lux >= self.lighting_thresholds["bright"]:
            return LightingCondition.BRIGHT
        elif lux >= self.lighting_thresholds["normal"]:
            return LightingCondition.NORMAL
        elif lux >= self.lighting_thresholds["low_light"]:
            return LightingCondition.LOW_LIGHT
        elif lux >= self.lighting_thresholds["dark"]:
            return LightingCondition.DARK
        else:
            return LightingCondition.COMPLETE_DARKNESS
    
    def _estimate_visibility(self, weather: WeatherCondition) -> float:
        """Estimate visibility in meters based on weather condition."""
        visibility_map = {
            WeatherCondition.CLEAR: 10000.0,
            WeatherCondition.RAIN: 5000.0,
            WeatherCondition.DUST_STORM: 100.0,
            WeatherCondition.FOG: 50.0,
            WeatherCondition.SNOW: 1000.0,
            WeatherCondition.EXTREME_HEAT: 8000.0
        }
        return visibility_map.get(weather, 10000.0)
    
    def _adapt_sensor_weights(self) -> None:
        """
        Adapt sensor weights based on current environmental conditions.
        
        Implements automatic switching based on lighting conditions (Requirement 2.4).
        """
        if not self.environmental_conditions:
            return
        
        conditions = self.environmental_conditions
        new_weights = {}
        
        # Adapt weights based on lighting condition
        if conditions.lighting == LightingCondition.COMPLETE_DARKNESS:
            # Complete darkness: prioritize thermal and infrared
            new_weights = {
                SensorType.THERMAL: 0.5,
                SensorType.INFRARED: 0.3,
                SensorType.RADAR: 0.2,
                SensorType.VISIBLE: 0.0
            }
        elif conditions.lighting == LightingCondition.DARK:
            # Dark conditions: thermal and infrared preferred
            new_weights = {
                SensorType.THERMAL: 0.4,
                SensorType.INFRARED: 0.3,
                SensorType.RADAR: 0.2,
                SensorType.VISIBLE: 0.1
            }
        elif conditions.lighting == LightingCondition.LOW_LIGHT:
            # Low light: balanced approach with thermal preference
            new_weights = {
                SensorType.THERMAL: 0.35,
                SensorType.VISIBLE: 0.25,
                SensorType.INFRARED: 0.25,
                SensorType.RADAR: 0.15
            }
        else:
            # Normal/bright conditions: visible camera preferred
            new_weights = {
                SensorType.VISIBLE: 0.5,
                SensorType.THERMAL: 0.25,
                SensorType.INFRARED: 0.15,
                SensorType.RADAR: 0.1
            }
        
        # Apply weather-based adjustments
        if conditions.weather in [WeatherCondition.DUST_STORM, WeatherCondition.FOG]:
            # Reduce visible camera weight in poor visibility
            if SensorType.VISIBLE in new_weights:
                visible_weight = new_weights[SensorType.VISIBLE]
                new_weights[SensorType.VISIBLE] = visible_weight * 0.5
                # Redistribute to thermal
                if SensorType.THERMAL in new_weights:
                    new_weights[SensorType.THERMAL] += visible_weight * 0.5
        
        # Update weights for active sensors only
        for sensor_type in self.active_sensors:
            if sensor_type in new_weights:
                self.sensor_weights[sensor_type] = new_weights[sensor_type]
        
        # Normalize weights
        total_weight = sum(self.sensor_weights.values())
        if total_weight > 0:
            for sensor_type in self.sensor_weights:
                self.sensor_weights[sensor_type] /= total_weight
        
        logger.debug(f"Adapted sensor weights: {self.sensor_weights}")
    
    def _apply_environmental_adaptation(self, detections: List[Detection]) -> List[Detection]:
        """Apply environmental adaptations to detection results."""
        if not self.environmental_conditions or not detections:
            return detections
        
        adapted_detections = []
        
        for detection in detections:
            adapted_detection = detection
            
            # Adjust confidence based on environmental conditions
            confidence_adjustment = self._calculate_environmental_confidence_adjustment()
            
            if confidence_adjustment != 0:
                new_confidence = max(0.0, min(1.0, detection.confidence + confidence_adjustment))
                
                adapted_detection = Detection(
                    id=detection.id,
                    camera_id=detection.camera_id,
                    timestamp=detection.timestamp,
                    bbox=detection.bbox,
                    confidence=new_confidence,
                    detection_class=detection.detection_class,
                    features=detection.features,
                    image_crop=detection.image_crop,
                    metadata={
                        **(detection.metadata or {}),
                        "environmental_adaptation": True,
                        "confidence_adjustment": confidence_adjustment,
                        "lighting_condition": self.environmental_conditions.lighting.value,
                        "weather_condition": self.environmental_conditions.weather.value
                    }
                )
            
            adapted_detections.append(adapted_detection)
        
        return adapted_detections
    
    def _calculate_environmental_confidence_adjustment(self) -> float:
        """Calculate confidence adjustment based on environmental conditions."""
        if not self.environmental_conditions:
            return 0.0
        
        adjustment = 0.0
        conditions = self.environmental_conditions
        
        # Lighting-based adjustments
        if conditions.lighting == LightingCondition.COMPLETE_DARKNESS:
            adjustment -= 0.1  # Slightly reduce confidence in complete darkness
        elif conditions.lighting == LightingCondition.BRIGHT:
            adjustment += 0.05  # Boost confidence in good lighting
        
        # Weather-based adjustments
        if conditions.weather in [WeatherCondition.DUST_STORM, WeatherCondition.FOG]:
            adjustment -= 0.15  # Reduce confidence in poor visibility
        elif conditions.weather == WeatherCondition.CLEAR:
            adjustment += 0.05  # Boost confidence in clear weather
        
        return max(-0.2, min(0.2, adjustment))  # Limit adjustment to ±0.2
    
    def get_active_sensors(self) -> List[str]:
        """Get list of currently active sensor types."""
        return [sensor.value for sensor in self.active_sensors]
    
    def calibrate_sensors(self, calibration_data: Dict[str, Any]) -> bool:
        """
        Calibrate sensor alignment and fusion parameters.
        
        Args:
            calibration_data: Calibration parameters for sensor alignment
            
        Returns:
            bool: True if calibration successful, False otherwise
        """
        try:
            # Update IoU threshold if provided
            if "iou_threshold" in calibration_data:
                self.iou_threshold = calibration_data["iou_threshold"]
            
            # Update confidence boost if provided
            if "confidence_boost" in calibration_data:
                self.confidence_boost = calibration_data["confidence_boost"]
            
            # Update distance threshold for radar correlation
            if "distance_threshold" in calibration_data:
                self.distance_threshold = calibration_data["distance_threshold"]
            
            # Update sensor weights if provided
            if "sensor_weights" in calibration_data:
                for sensor_name, weight in calibration_data["sensor_weights"].items():
                    try:
                        sensor_type = SensorType(sensor_name)
                        if sensor_type in self.active_sensors:
                            self.sensor_weights[sensor_type] = weight
                    except ValueError:
                        logger.warning(f"Unknown sensor type in calibration: {sensor_name}")
            
            # Update lighting thresholds if provided
            if "lighting_thresholds" in calibration_data:
                self.lighting_thresholds.update(calibration_data["lighting_thresholds"])
            
            logger.info("Sensor calibration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Sensor calibration failed: {e}")
            return False