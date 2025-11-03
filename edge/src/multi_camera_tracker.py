"""
Multi-camera tracking implementation using ByteTrack algorithm with Kalman filtering.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import uuid
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from loguru import logger

from shared.interfaces.tracking import IMultiCameraTracker
from shared.models import Detection, Track, GlobalTrack, Trajectory, Point, TrackStatus


class KalmanTracker:
    """Kalman filter for tracking individual objects."""
    
    def __init__(self, detection: Detection):
        """Initialize Kalman tracker with first detection."""
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State vector: [x, y, w, h, dx, dy, dw, dh]
        # Measurement vector: [x, y, w, h]
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Process noise covariance
        self.kf.Q *= 0.01
        
        # Measurement noise covariance
        self.kf.R *= 0.1
        
        # Initial state covariance
        self.kf.P *= 1000
        
        # Initialize state with first detection
        bbox = detection.bbox
        self.kf.x = np.array([bbox.x, bbox.y, bbox.width, bbox.height, 0, 0, 0, 0])
        
        self.time_since_update = 0
        self.hit_streak = 1
        self.age = 1
        
    def update(self, detection: Detection) -> None:
        """Update tracker with new detection."""
        self.time_since_update = 0
        self.hit_streak += 1
        self.age += 1
        
        bbox = detection.bbox
        measurement = np.array([bbox.x, bbox.y, bbox.width, bbox.height])
        self.kf.update(measurement)
        
    def predict(self) -> np.ndarray:
        """Predict next state and return predicted bounding box."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0
        if self.kf.x[7] + self.kf.x[3] <= 0:
            self.kf.x[7] = 0
            
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        return self.kf.x[:4]
    
    def get_state(self) -> np.ndarray:
        """Get current state as bounding box."""
        return self.kf.x[:4]


class MultiCameraTracker(IMultiCameraTracker):
    """Multi-camera tracking system using ByteTrack algorithm."""
    
    def __init__(self, 
                 max_disappeared: int = 30,
                 max_distance: float = 100.0,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3):
        """
        Initialize multi-camera tracker.
        
        Args:
            max_disappeared: Maximum frames a track can be missing before deletion
            max_distance: Maximum distance for track association
            min_hits: Minimum hits before track is considered confirmed
            iou_threshold: IoU threshold for track association
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        # Track storage per camera
        self.camera_trackers: Dict[str, Dict[str, KalmanTracker]] = defaultdict(dict)
        self.camera_tracks: Dict[str, Dict[str, Track]] = defaultdict(dict)
        self.global_tracks: Dict[str, GlobalTrack] = {}
        
        # Track ID counters
        self.next_track_id = 1
        self.next_global_id = 1
        
        logger.info("MultiCameraTracker initialized")
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _associate_detections_to_tracks(self, 
                                      detections: List[Detection], 
                                      trackers: Dict[str, KalmanTracker]) -> Tuple[List[Tuple[int, str]], List[int], List[str]]:
        """Associate detections to existing tracks using Hungarian algorithm."""
        if not trackers:
            return [], list(range(len(detections))), []
        
        # Get predictions from all trackers
        predictions = {}
        for track_id, tracker in trackers.items():
            predictions[track_id] = tracker.predict()
        
        # Calculate cost matrix (1 - IoU)
        cost_matrix = np.zeros((len(detections), len(trackers)))
        tracker_ids = list(trackers.keys())
        
        for d, detection in enumerate(detections):
            det_box = np.array([detection.bbox.x, detection.bbox.y, 
                              detection.bbox.width, detection.bbox.height])
            
            for t, track_id in enumerate(tracker_ids):
                pred_box = predictions[track_id]
                iou = self._calculate_iou(det_box, pred_box)
                cost_matrix[d, t] = 1 - iou
        
        # Apply Hungarian algorithm
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Filter matches based on IoU threshold
            matches = []
            unmatched_detections = list(range(len(detections)))
            unmatched_trackers = list(range(len(trackers)))
            
            for r, c in zip(row_indices, col_indices):
                if cost_matrix[r, c] <= (1 - self.iou_threshold):
                    matches.append((r, tracker_ids[c]))
                    unmatched_detections.remove(r)
                    unmatched_trackers.remove(c)
            
            unmatched_tracker_ids = [tracker_ids[i] for i in unmatched_trackers]
            
            return matches, unmatched_detections, unmatched_tracker_ids
        
        return [], list(range(len(detections))), list(trackers.keys())
    
    def update_tracks(self, detections: List[Detection], camera_id: str) -> List[Track]:
        """Update tracks with new detections from a camera."""
        logger.debug(f"Updating tracks for camera {camera_id} with {len(detections)} detections")
        
        # Get existing trackers for this camera
        trackers = self.camera_trackers[camera_id]
        tracks = self.camera_tracks[camera_id]
        
        # Associate detections to tracks
        matches, unmatched_detections, unmatched_trackers = self._associate_detections_to_tracks(
            detections, trackers
        )
        
        # Update matched tracks
        for det_idx, track_id in matches:
            detection = detections[det_idx]
            trackers[track_id].update(detection)
            tracks[track_id].add_detection(detection)
            tracks[track_id].status = TrackStatus.ACTIVE
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            track_id = f"{camera_id}_{self.next_track_id}"
            self.next_track_id += 1
            
            # Create Kalman tracker
            kalman_tracker = KalmanTracker(detection)
            trackers[track_id] = kalman_tracker
            
            # Create track object
            track = Track(
                id=track_id,
                camera_id=camera_id,
                start_time=detection.timestamp
            )
            track.add_detection(detection)
            tracks[track_id] = track
            
            logger.debug(f"Created new track {track_id} for camera {camera_id}")
        
        # Handle unmatched tracks (mark as lost or remove)
        tracks_to_remove = []
        for track_id in unmatched_trackers:
            tracker = trackers[track_id]
            track = tracks[track_id]
            
            # Predict next position
            tracker.predict()
            
            if tracker.time_since_update > self.max_disappeared:
                tracks_to_remove.append(track_id)
                track.status = TrackStatus.LOST
                track.end_time = datetime.now()
                logger.debug(f"Removing lost track {track_id}")
            else:
                track.status = TrackStatus.LOST
        
        # Remove old tracks
        for track_id in tracks_to_remove:
            del trackers[track_id]
            del tracks[track_id]
        
        # Return active tracks
        active_tracks = [track for track in tracks.values() 
                        if track.status == TrackStatus.ACTIVE and len(track.detections) >= self.min_hits]
        
        logger.debug(f"Camera {camera_id} has {len(active_tracks)} active tracks")
        return active_tracks
    
    def match_across_cameras(self, track: Track) -> Optional[GlobalTrack]:
        """Match a local track to a global track across cameras."""
        # This is a simplified implementation
        # In a full implementation, this would use re-identification features
        
        if not track.detections:
            return None
        
        # For now, create a new global track for each local track
        # This should be enhanced with proper re-identification matching
        global_track_id = f"global_{self.next_global_id}"
        self.next_global_id += 1
        
        global_track = GlobalTrack(
            id=global_track_id,
            start_time=track.start_time
        )
        global_track.add_local_track(track, 1.0)  # Perfect confidence for now
        
        self.global_tracks[global_track_id] = global_track
        
        logger.debug(f"Created global track {global_track_id} for local track {track.id}")
        return global_track
    
    def get_trajectory(self, track_id: str) -> Optional[Trajectory]:
        """Get trajectory for a specific track."""
        # Check local tracks first
        for camera_tracks in self.camera_tracks.values():
            if track_id in camera_tracks:
                return camera_tracks[track_id].trajectory
        
        # Check global tracks
        if track_id in self.global_tracks:
            global_track = self.global_tracks[track_id]
            if global_track.local_tracks:
                # Combine trajectories from all local tracks
                combined_points = []
                for local_track in global_track.local_tracks:
                    combined_points.extend(local_track.trajectory.points)
                
                # Sort by timestamp
                combined_points.sort(key=lambda p: p.timestamp or datetime.min)
                
                if combined_points:
                    trajectory = Trajectory(
                        points=combined_points,
                        start_time=combined_points[0].timestamp or datetime.now(),
                        end_time=combined_points[-1].timestamp
                    )
                    return trajectory
        
        return None
    
    def get_active_tracks(self, camera_id: Optional[str] = None) -> List[Track]:
        """Get all active tracks, optionally filtered by camera."""
        active_tracks = []
        
        if camera_id:
            # Return tracks for specific camera
            camera_tracks = self.camera_tracks.get(camera_id, {})
            active_tracks = [track for track in camera_tracks.values() 
                           if track.status == TrackStatus.ACTIVE]
        else:
            # Return tracks for all cameras
            for camera_tracks in self.camera_tracks.values():
                active_tracks.extend([track for track in camera_tracks.values() 
                                    if track.status == TrackStatus.ACTIVE])
        
        return active_tracks
    
    def cleanup_old_tracks(self, max_age_seconds: int) -> int:
        """Remove old inactive tracks and return count removed."""
        removed_count = 0
        cutoff_time = datetime.now() - timedelta(seconds=max_age_seconds)
        
        # Clean up local tracks
        for camera_id in list(self.camera_tracks.keys()):
            tracks_to_remove = []
            for track_id, track in self.camera_tracks[camera_id].items():
                if (track.status != TrackStatus.ACTIVE and 
                    track.end_time and track.end_time < cutoff_time):
                    tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                del self.camera_tracks[camera_id][track_id]
                if track_id in self.camera_trackers[camera_id]:
                    del self.camera_trackers[camera_id][track_id]
                removed_count += 1
        
        # Clean up global tracks
        global_tracks_to_remove = []
        for global_id, global_track in self.global_tracks.items():
            if (global_track.status != TrackStatus.ACTIVE and 
                global_track.end_time and global_track.end_time < cutoff_time):
                global_tracks_to_remove.append(global_id)
        
        for global_id in global_tracks_to_remove:
            del self.global_tracks[global_id]
            removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old tracks")
        
        return removed_count