"""
Drone Tracking and Evidence Collection Module
Handles real-time drone tracking and evidence collection capabilities
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import uuid

logger = logging.getLogger(__name__)

class EvidenceType(str, Enum):
    VIDEO = "video"
    PHOTO = "photo"
    THERMAL_IMAGE = "thermal_image"
    GPS_TRACK = "gps_track"
    SENSOR_DATA = "sensor_data"

@dataclass
class GPSCoordinate:
    latitude: float
    longitude: float
    altitude: float
    timestamp: datetime
    accuracy: float

@dataclass
class EvidenceItem:
    id: str
    type: EvidenceType
    timestamp: datetime
    location: GPSCoordinate
    file_path: Optional[str]
    metadata: Dict
    drone_id: str
    deployment_id: str

@dataclass
class FlightPath:
    waypoints: List[GPSCoordinate]
    start_time: datetime
    end_time: Optional[datetime]
    total_distance: float
    max_altitude: float

class DroneTracker:
    """Real-time drone tracking and flight path management"""
    
    def __init__(self):
        self.active_tracks: Dict[str, FlightPath] = {}
        self.evidence_items: Dict[str, List[EvidenceItem]] = {}
        self.tracking_interval = 5  # seconds
        
    async def start_tracking(self, drone_id: str, deployment_id: str, initial_location: GPSCoordinate):
        """Start tracking a drone's flight path"""
        flight_path = FlightPath(
            waypoints=[initial_location],
            start_time=datetime.now(),
            end_time=None,
            total_distance=0.0,
            max_altitude=initial_location.altitude
        )
        
        self.active_tracks[drone_id] = flight_path
        self.evidence_items[deployment_id] = []
        
        logger.info(f"Started tracking drone {drone_id} for deployment {deployment_id}")
        
        # Start continuous tracking
        asyncio.create_task(self._continuous_tracking(drone_id))
    
    async def _continuous_tracking(self, drone_id: str):
        """Continuously track drone position"""
        while drone_id in self.active_tracks:
            try:
                # Simulate getting GPS data from drone
                current_location = await self._get_drone_location(drone_id)
                if current_location:
                    await self.update_position(drone_id, current_location)
                
                await asyncio.sleep(self.tracking_interval)
                
            except Exception as e:
                logger.error(f"Error tracking drone {drone_id}: {e}")
                await asyncio.sleep(self.tracking_interval)
    
    async def update_position(self, drone_id: str, location: GPSCoordinate):
        """Update drone position and calculate flight metrics"""
        if drone_id not in self.active_tracks:
            return
        
        flight_path = self.active_tracks[drone_id]
        
        # Add waypoint
        flight_path.waypoints.append(location)
        
        # Update metrics
        if len(flight_path.waypoints) > 1:
            last_point = flight_path.waypoints[-2]
            distance = self._calculate_distance(last_point, location)
            flight_path.total_distance += distance
        
        flight_path.max_altitude = max(flight_path.max_altitude, location.altitude)
        
        logger.debug(f"Updated position for drone {drone_id}: {location.latitude}, {location.longitude}")
    
    async def stop_tracking(self, drone_id: str):
        """Stop tracking a drone"""
        if drone_id in self.active_tracks:
            flight_path = self.active_tracks[drone_id]
            flight_path.end_time = datetime.now()
            
            logger.info(f"Stopped tracking drone {drone_id}. Total distance: {flight_path.total_distance:.2f}m")
            
            # Archive the flight path
            del self.active_tracks[drone_id]
    
    async def _get_drone_location(self, drone_id: str) -> Optional[GPSCoordinate]:
        """Get current drone location (simulated)"""
        # In real implementation, this would communicate with drone hardware
        # For now, simulate movement
        if drone_id in self.active_tracks:
            last_location = self.active_tracks[drone_id].waypoints[-1]
            
            # Simulate small movement
            new_location = GPSCoordinate(
                latitude=last_location.latitude + (0.0001 * (0.5 - asyncio.get_event_loop().time() % 1)),
                longitude=last_location.longitude + (0.0001 * (0.5 - asyncio.get_event_loop().time() % 1)),
                altitude=last_location.altitude + (5 * (0.5 - asyncio.get_event_loop().time() % 1)),
                timestamp=datetime.now(),
                accuracy=3.0
            )
            
            return new_location
        
        return None
    
    def _calculate_distance(self, point1: GPSCoordinate, point2: GPSCoordinate) -> float:
        """Calculate distance between two GPS points in meters"""
        # Simplified distance calculation (Haversine formula would be more accurate)
        lat_diff = point2.latitude - point1.latitude
        lon_diff = point2.longitude - point1.longitude
        alt_diff = point2.altitude - point1.altitude
        
        # Convert to meters (approximate)
        lat_meters = lat_diff * 111000  # 1 degree ≈ 111km
        lon_meters = lon_diff * 111000 * abs(point1.latitude / 90)  # Adjust for latitude
        
        return (lat_meters**2 + lon_meters**2 + alt_diff**2)**0.5
    
    def get_flight_path(self, drone_id: str) -> Optional[FlightPath]:
        """Get current flight path for a drone"""
        return self.active_tracks.get(drone_id)
    
    def get_evidence_items(self, deployment_id: str) -> List[EvidenceItem]:
        """Get all evidence items for a deployment"""
        return self.evidence_items.get(deployment_id, [])

class EvidenceCollector:
    """Handles evidence collection from drones"""
    
    def __init__(self, tracker: DroneTracker):
        self.tracker = tracker
        self.collection_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task = None
    
    async def start_collection_processor(self):
        """Start the evidence collection processor"""
        self.processing_task = asyncio.create_task(self._process_evidence_queue())
    
    async def stop_collection_processor(self):
        """Stop the evidence collection processor"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
    
    async def collect_evidence(
        self, 
        drone_id: str, 
        deployment_id: str, 
        evidence_type: EvidenceType,
        metadata: Optional[Dict] = None
    ) -> str:
        """Collect evidence item from drone"""
        
        # Get current drone location
        flight_path = self.tracker.get_flight_path(drone_id)
        if not flight_path or not flight_path.waypoints:
            raise ValueError(f"No tracking data available for drone {drone_id}")
        
        current_location = flight_path.waypoints[-1]
        
        # Create evidence item
        evidence_id = str(uuid.uuid4())
        evidence_item = EvidenceItem(
            id=evidence_id,
            type=evidence_type,
            timestamp=datetime.now(),
            location=current_location,
            file_path=None,  # Would be set after file processing
            metadata=metadata or {},
            drone_id=drone_id,
            deployment_id=deployment_id
        )
        
        # Add to deployment evidence
        if deployment_id not in self.tracker.evidence_items:
            self.tracker.evidence_items[deployment_id] = []
        
        self.tracker.evidence_items[deployment_id].append(evidence_item)
        
        # Queue for processing
        await self.collection_queue.put(evidence_item)
        
        logger.info(f"Evidence collected: {evidence_id} ({evidence_type}) from drone {drone_id}")
        
        return evidence_id
    
    async def _process_evidence_queue(self):
        """Process evidence collection queue"""
        while True:
            try:
                evidence_item = await self.collection_queue.get()
                await self._process_evidence_item(evidence_item)
                self.collection_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing evidence item: {e}")
    
    async def _process_evidence_item(self, evidence_item: EvidenceItem):
        """Process individual evidence item"""
        try:
            # Simulate evidence processing based on type
            if evidence_item.type == EvidenceType.VIDEO:
                await self._process_video_evidence(evidence_item)
            elif evidence_item.type == EvidenceType.PHOTO:
                await self._process_photo_evidence(evidence_item)
            elif evidence_item.type == EvidenceType.THERMAL_IMAGE:
                await self._process_thermal_evidence(evidence_item)
            elif evidence_item.type == EvidenceType.GPS_TRACK:
                await self._process_gps_evidence(evidence_item)
            elif evidence_item.type == EvidenceType.SENSOR_DATA:
                await self._process_sensor_evidence(evidence_item)
            
            logger.info(f"Processed evidence item {evidence_item.id}")
            
        except Exception as e:
            logger.error(f"Failed to process evidence item {evidence_item.id}: {e}")
    
    async def _process_video_evidence(self, evidence_item: EvidenceItem):
        """Process video evidence"""
        # Simulate video processing
        await asyncio.sleep(0.5)
        evidence_item.file_path = f"/evidence/video/{evidence_item.id}.mp4"
        evidence_item.metadata.update({
            "duration": 30,
            "resolution": "1920x1080",
            "codec": "h264"
        })
    
    async def _process_photo_evidence(self, evidence_item: EvidenceItem):
        """Process photo evidence"""
        await asyncio.sleep(0.2)
        evidence_item.file_path = f"/evidence/photos/{evidence_item.id}.jpg"
        evidence_item.metadata.update({
            "resolution": "4096x3072",
            "format": "jpeg"
        })
    
    async def _process_thermal_evidence(self, evidence_item: EvidenceItem):
        """Process thermal image evidence"""
        await asyncio.sleep(0.3)
        evidence_item.file_path = f"/evidence/thermal/{evidence_item.id}.tiff"
        evidence_item.metadata.update({
            "temperature_range": {"min": 15.2, "max": 37.8},
            "format": "tiff"
        })
    
    async def _process_gps_evidence(self, evidence_item: EvidenceItem):
        """Process GPS track evidence"""
        await asyncio.sleep(0.1)
        evidence_item.file_path = f"/evidence/gps/{evidence_item.id}.gpx"
        evidence_item.metadata.update({
            "format": "gpx",
            "waypoint_count": len(self.tracker.get_flight_path(evidence_item.drone_id).waypoints)
        })
    
    async def _process_sensor_evidence(self, evidence_item: EvidenceItem):
        """Process sensor data evidence"""
        await asyncio.sleep(0.1)
        evidence_item.file_path = f"/evidence/sensors/{evidence_item.id}.json"
        evidence_item.metadata.update({
            "sensors": ["accelerometer", "gyroscope", "magnetometer"],
            "format": "json"
        })
    
    async def get_evidence_summary(self, deployment_id: str) -> Dict:
        """Get evidence collection summary for deployment"""
        evidence_items = self.tracker.get_evidence_items(deployment_id)
        
        summary = {
            "total_items": len(evidence_items),
            "by_type": {},
            "collection_timespan": None,
            "total_size_estimate": 0
        }
        
        if evidence_items:
            # Count by type
            for item in evidence_items:
                item_type = item.type.value
                summary["by_type"][item_type] = summary["by_type"].get(item_type, 0) + 1
            
            # Calculate timespan
            timestamps = [item.timestamp for item in evidence_items]
            summary["collection_timespan"] = {
                "start": min(timestamps),
                "end": max(timestamps),
                "duration_minutes": (max(timestamps) - min(timestamps)).total_seconds() / 60
            }
            
            # Estimate total size (simplified)
            size_estimates = {
                EvidenceType.VIDEO: 50,  # MB
                EvidenceType.PHOTO: 5,   # MB
                EvidenceType.THERMAL_IMAGE: 2,  # MB
                EvidenceType.GPS_TRACK: 0.1,    # MB
                EvidenceType.SENSOR_DATA: 0.5   # MB
            }
            
            for item in evidence_items:
                summary["total_size_estimate"] += size_estimates.get(item.type, 1)
        
        return summary