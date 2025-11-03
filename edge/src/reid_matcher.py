"""
Person re-identification engine using OSNet/FastReID models.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import os
from loguru import logger

from shared.interfaces.tracking import IReIDMatcher


class OSNetFeatureExtractor:
    """Feature extractor using OSNet architecture."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize OSNet feature extractor.
        
        Args:
            model_path: Path to pre-trained OSNet model
            device: Device to run inference on
        """
        self.device = device
        self.input_size = (256, 128)  # Standard ReID input size
        self.model = None
        
        # For now, we'll use a simple CNN-based feature extractor
        # In production, this would load a pre-trained OSNet model
        self._init_simple_model()
        
        logger.info(f"OSNet feature extractor initialized on {device}")
    
    def _init_simple_model(self):
        """Initialize a simple CNN model for feature extraction."""
        # This is a simplified model for demonstration
        # In production, use pre-trained OSNet or FastReID models
        import torch.nn as nn
        
        class SimpleReIDModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Linear(256, 512)
                
            def forward(self, x):
                x = self.backbone(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return F.normalize(x, p=2, dim=1)
        
        self.model = SimpleReIDModel()
        if torch.cuda.is_available() and self.device == "cuda":
            self.model = self.model.cuda()
        self.model.eval()
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for feature extraction."""
        # Resize to standard ReID input size
        image = cv2.resize(image, self.input_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to tensor and add batch dimension
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        
        if torch.cuda.is_available() and self.device == "cuda":
            image = image.cuda()
        
        return image
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract feature embedding from person image."""
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            features = self.model(input_tensor)
            return features.cpu().numpy().flatten()


class ReIDMatcher(IReIDMatcher):
    """Person re-identification matcher with feature gallery management."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 gallery_path: Optional[str] = None,
                 max_gallery_size: int = 10000):
        """
        Initialize ReID matcher.
        
        Args:
            model_path: Path to pre-trained ReID model
            device: Device to run inference on
            gallery_path: Path to save/load gallery
            max_gallery_size: Maximum number of persons in gallery
        """
        self.device = device
        self.gallery_path = gallery_path or "reid_gallery.pkl"
        self.max_gallery_size = max_gallery_size
        
        # Initialize feature extractor
        self.feature_extractor = OSNetFeatureExtractor(model_path, device)
        
        # Gallery storage: {person_id: (features, timestamp, update_count)}
        self.gallery: Dict[str, Tuple[np.ndarray, datetime, int]] = {}
        
        # Load existing gallery if available
        self._load_gallery()
        
        logger.info(f"ReIDMatcher initialized with gallery size: {len(self.gallery)}")
    
    def _load_gallery(self) -> None:
        """Load gallery from disk."""
        if os.path.exists(self.gallery_path):
            try:
                with open(self.gallery_path, 'rb') as f:
                    self.gallery = pickle.load(f)
                logger.info(f"Loaded gallery with {len(self.gallery)} persons")
            except Exception as e:
                logger.warning(f"Failed to load gallery: {e}")
                self.gallery = {}
    
    def _save_gallery(self) -> None:
        """Save gallery to disk."""
        try:
            with open(self.gallery_path, 'wb') as f:
                pickle.dump(self.gallery, f)
            logger.debug(f"Saved gallery with {len(self.gallery)} persons")
        except Exception as e:
            logger.error(f"Failed to save gallery: {e}")
    
    def _calculate_cosine_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors."""
        # Normalize features
        features1 = features1 / (np.linalg.norm(features1) + 1e-8)
        features2 = features2 / (np.linalg.norm(features2) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(features1, features2)
        return float(similarity)
    
    def extract_features(self, person_crop: np.ndarray) -> np.ndarray:
        """Extract feature embedding from person image crop."""
        if person_crop is None or person_crop.size == 0:
            logger.warning("Empty person crop provided")
            return np.zeros(512)  # Return zero vector
        
        try:
            features = self.feature_extractor.extract_features(person_crop)
            return features
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(512)
    
    def match_person(self, 
                    features: np.ndarray, 
                    gallery: Optional[List[Tuple[str, np.ndarray]]] = None,
                    threshold: float = 0.7) -> Optional[Tuple[str, float]]:
        """
        Match person features against gallery.
        
        Args:
            features: Query features
            gallery: Optional external gallery, uses internal if None
            threshold: Similarity threshold for matching
            
        Returns:
            Tuple of (person_id, confidence) if match found, None otherwise
        """
        if gallery is None:
            # Use internal gallery
            gallery = [(pid, data[0]) for pid, data in self.gallery.items()]
        
        if not gallery:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for person_id, gallery_features in gallery:
            similarity = self._calculate_cosine_similarity(features, gallery_features)
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = person_id
        
        if best_match:
            logger.debug(f"Matched person {best_match} with confidence {best_similarity:.3f}")
            return (best_match, best_similarity)
        
        return None
    
    def add_to_gallery(self, person_id: str, features: np.ndarray) -> None:
        """Add person features to the gallery."""
        if len(self.gallery) >= self.max_gallery_size:
            # Remove oldest entry
            oldest_id = min(self.gallery.keys(), 
                          key=lambda x: self.gallery[x][1])
            del self.gallery[oldest_id]
            logger.debug(f"Removed oldest gallery entry: {oldest_id}")
        
        self.gallery[person_id] = (features, datetime.now(), 1)
        self._save_gallery()
        
        logger.debug(f"Added person {person_id} to gallery")
    
    def update_gallery(self, person_id: str, new_features: np.ndarray) -> None:
        """Update existing person features in gallery using exponential moving average."""
        if person_id not in self.gallery:
            self.add_to_gallery(person_id, new_features)
            return
        
        old_features, timestamp, update_count = self.gallery[person_id]
        
        # Exponential moving average with adaptive alpha
        alpha = min(0.3, 1.0 / (update_count + 1))
        updated_features = (1 - alpha) * old_features + alpha * new_features
        
        # Normalize updated features
        updated_features = updated_features / (np.linalg.norm(updated_features) + 1e-8)
        
        self.gallery[person_id] = (updated_features, datetime.now(), update_count + 1)
        self._save_gallery()
        
        logger.debug(f"Updated person {person_id} in gallery (update #{update_count + 1})")
    
    def get_gallery_size(self) -> int:
        """Get number of persons in the gallery."""
        return len(self.gallery)
    
    def cleanup_gallery(self, max_age_days: int) -> int:
        """Remove old entries from gallery and return count removed."""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        old_entries = [pid for pid, (_, timestamp, _) in self.gallery.items() 
                      if timestamp < cutoff_time]
        
        for pid in old_entries:
            del self.gallery[pid]
        
        if old_entries:
            self._save_gallery()
            logger.info(f"Cleaned up {len(old_entries)} old gallery entries")
        
        return len(old_entries)
    
    def get_gallery_stats(self) -> Dict[str, Any]:
        """Get gallery statistics."""
        if not self.gallery:
            return {"size": 0, "avg_updates": 0, "oldest_entry": None, "newest_entry": None}
        
        timestamps = [data[1] for data in self.gallery.values()]
        update_counts = [data[2] for data in self.gallery.values()]
        
        return {
            "size": len(self.gallery),
            "avg_updates": sum(update_counts) / len(update_counts),
            "oldest_entry": min(timestamps),
            "newest_entry": max(timestamps)
        }
    
    def export_gallery(self, export_path: str) -> bool:
        """Export gallery to specified path."""
        try:
            with open(export_path, 'wb') as f:
                pickle.dump(self.gallery, f)
            logger.info(f"Exported gallery to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export gallery: {e}")
            return False
    
    def import_gallery(self, import_path: str, merge: bool = True) -> bool:
        """Import gallery from specified path."""
        try:
            with open(import_path, 'rb') as f:
                imported_gallery = pickle.load(f)
            
            if merge:
                # Merge with existing gallery
                for person_id, data in imported_gallery.items():
                    if person_id in self.gallery:
                        # Keep the entry with more updates
                        if data[2] > self.gallery[person_id][2]:
                            self.gallery[person_id] = data
                    else:
                        self.gallery[person_id] = data
            else:
                # Replace existing gallery
                self.gallery = imported_gallery
            
            self._save_gallery()
            logger.info(f"Imported gallery from {import_path} (merge={merge})")
            return True
        except Exception as e:
            logger.error(f"Failed to import gallery: {e}")
            return False