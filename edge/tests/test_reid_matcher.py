"""
Unit tests for Person Re-Identification system.

Tests feature extraction, person matching, gallery management,
and re-identification performance across different conditions.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from edge.src.reid_matcher import ReIDMatcher, OSNetFeatureExtractor


class TestOSNetFeatureExtractor:
    """Test cases for OSNetFeatureExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = OSNetFeatureExtractor(device="cpu")  # Use CPU for testing
    
    def test_feature_extractor_initialization(self):
        """Test OSNetFeatureExtractor initializes correctly."""
        assert self.extractor.device == "cpu"
        assert self.extractor.input_size == (256, 128)
        assert self.extractor.model is not None
    
    def test_preprocess_image(self):
        """Test image preprocessing for feature extraction."""
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        preprocessed = self.extractor.preprocess_image(test_image)
        
        assert preprocessed.shape == (1, 3, 128, 256)  # Batch, channels, height, width
        assert preprocessed.dtype == np.float32 if hasattr(preprocessed, 'dtype') else True
    
    def test_extract_features(self):
        """Test feature extraction from person image."""
        # Create test person crop
        person_crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        
        features = self.extractor.extract_features(person_crop)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (512,)  # Expected feature dimension
        assert not np.allclose(features, 0)  # Features should not be all zeros
    
    def test_extract_features_different_sizes(self):
        """Test feature extraction with different input sizes."""
        # Test with different sized crops
        sizes = [(100, 50, 3), (300, 150, 3), (200, 100, 3)]
        
        for size in sizes:
            person_crop = np.random.randint(0, 255, size, dtype=np.uint8)
            features = self.extractor.extract_features(person_crop)
            
            assert features.shape == (512,)
            assert not np.allclose(features, 0)


class TestReIDMatcher:
    """Test cases for ReIDMatcher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary file for gallery
        self.temp_gallery = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
        self.temp_gallery.close()
        
        self.matcher = ReIDMatcher(
            device="cpu",
            gallery_path=self.temp_gallery.name,
            max_gallery_size=100
        )
        
        # Create test features
        self.test_features_1 = np.random.rand(512).astype(np.float32)
        self.test_features_2 = np.random.rand(512).astype(np.float32)
        self.test_features_similar = self.test_features_1 + np.random.rand(512) * 0.1
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_gallery.name):
            os.unlink(self.temp_gallery.name)
    
    def test_reid_matcher_initialization(self):
        """Test ReIDMatcher initializes correctly."""
        assert self.matcher.device == "cpu"
        assert self.matcher.gallery_path == self.temp_gallery.name
        assert self.matcher.max_gallery_size == 100
        assert len(self.matcher.gallery) == 0
    
    def test_calculate_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Test identical features
        similarity_identical = self.matcher._calculate_cosine_similarity(
            self.test_features_1, self.test_features_1
        )
        assert abs(similarity_identical - 1.0) < 1e-6
        
        # Test different features
        similarity_different = self.matcher._calculate_cosine_similarity(
            self.test_features_1, self.test_features_2
        )
        assert -1.0 <= similarity_different <= 1.0
        
        # Test similar features
        similarity_similar = self.matcher._calculate_cosine_similarity(
            self.test_features_1, self.test_features_similar
        )
        assert similarity_similar > similarity_different
    
    def test_extract_features(self):
        """Test feature extraction wrapper."""
        person_crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        
        features = self.matcher.extract_features(person_crop)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (512,)
    
    def test_extract_features_empty_crop(self):
        """Test feature extraction with empty crop."""
        empty_crop = np.array([])
        
        features = self.matcher.extract_features(empty_crop)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (512,)
        assert np.allclose(features, 0)  # Should return zero vector
    
    def test_add_to_gallery(self):
        """Test adding person to gallery."""
        person_id = "person_001"
        
        self.matcher.add_to_gallery(person_id, self.test_features_1)
        
        assert len(self.matcher.gallery) == 1
        assert person_id in self.matcher.gallery
        
        stored_features, timestamp, update_count = self.matcher.gallery[person_id]
        assert np.array_equal(stored_features, self.test_features_1)
        assert update_count == 1
    
    def test_update_gallery(self):
        """Test updating existing person in gallery."""
        person_id = "person_001"
        
        # Add initial features
        self.matcher.add_to_gallery(person_id, self.test_features_1)
        
        # Update with new features
        self.matcher.update_gallery(person_id, self.test_features_2)
        
        assert len(self.matcher.gallery) == 1
        stored_features, timestamp, update_count = self.matcher.gallery[person_id]
        
        # Features should be updated (exponential moving average)
        assert not np.array_equal(stored_features, self.test_features_1)
        assert not np.array_equal(stored_features, self.test_features_2)
        assert update_count == 2
    
    def test_match_person_no_gallery(self):
        """Test person matching with empty gallery."""
        result = self.matcher.match_person(self.test_features_1)
        
        assert result is None
    
    def test_match_person_with_gallery(self):
        """Test person matching with populated gallery."""
        # Add persons to gallery
        self.matcher.add_to_gallery("person_001", self.test_features_1)
        self.matcher.add_to_gallery("person_002", self.test_features_2)
        
        # Test matching with similar features
        result = self.matcher.match_person(self.test_features_similar, threshold=0.5)
        
        assert result is not None
        person_id, confidence = result
        assert person_id == "person_001"  # Should match the similar features
        assert 0.5 <= confidence <= 1.0
    
    def test_match_person_no_match(self):
        """Test person matching when no match meets threshold."""
        # Add person to gallery
        self.matcher.add_to_gallery("person_001", self.test_features_1)
        
        # Test matching with very different features and high threshold
        very_different_features = -self.test_features_1  # Opposite direction
        result = self.matcher.match_person(very_different_features, threshold=0.9)
        
        assert result is None
    
    def test_match_person_external_gallery(self):
        """Test person matching with external gallery."""
        external_gallery = [
            ("ext_person_001", self.test_features_1),
            ("ext_person_002", self.test_features_2)
        ]
        
        result = self.matcher.match_person(
            self.test_features_similar, 
            gallery=external_gallery, 
            threshold=0.5
        )
        
        assert result is not None
        person_id, confidence = result
        assert person_id == "ext_person_001"
    
    def test_gallery_size_limit(self):
        """Test gallery size limit enforcement."""
        # Set small gallery size for testing
        small_matcher = ReIDMatcher(
            device="cpu",
            gallery_path=self.temp_gallery.name + "_small",
            max_gallery_size=3
        )
        
        # Add more persons than the limit
        for i in range(5):
            features = np.random.rand(512).astype(np.float32)
            small_matcher.add_to_gallery(f"person_{i:03d}", features)
        
        # Gallery should not exceed the limit
        assert len(small_matcher.gallery) <= 3
        
        # Clean up
        if os.path.exists(small_matcher.gallery_path):
            os.unlink(small_matcher.gallery_path)
    
    def test_get_gallery_size(self):
        """Test getting gallery size."""
        assert self.matcher.get_gallery_size() == 0
        
        self.matcher.add_to_gallery("person_001", self.test_features_1)
        assert self.matcher.get_gallery_size() == 1
        
        self.matcher.add_to_gallery("person_002", self.test_features_2)
        assert self.matcher.get_gallery_size() == 2
    
    def test_cleanup_gallery(self):
        """Test cleanup of old gallery entries."""
        # Add person with old timestamp
        person_id = "old_person"
        old_timestamp = datetime.now() - timedelta(days=10)
        self.matcher.gallery[person_id] = (self.test_features_1, old_timestamp, 1)
        
        # Add recent person
        self.matcher.add_to_gallery("recent_person", self.test_features_2)
        
        # Cleanup entries older than 5 days
        removed_count = self.matcher.cleanup_gallery(max_age_days=5)
        
        assert removed_count == 1
        assert "old_person" not in self.matcher.gallery
        assert "recent_person" in self.matcher.gallery
    
    def test_get_gallery_stats(self):
        """Test getting gallery statistics."""
        # Empty gallery
        stats = self.matcher.get_gallery_stats()
        assert stats["size"] == 0
        
        # Add some persons
        self.matcher.add_to_gallery("person_001", self.test_features_1)
        self.matcher.update_gallery("person_001", self.test_features_2)
        self.matcher.add_to_gallery("person_002", self.test_features_2)
        
        stats = self.matcher.get_gallery_stats()
        assert stats["size"] == 2
        assert stats["avg_updates"] == 1.5  # (2 + 1) / 2
        assert stats["oldest_entry"] is not None
        assert stats["newest_entry"] is not None
    
    def test_export_import_gallery(self):
        """Test gallery export and import functionality."""
        # Add persons to gallery
        self.matcher.add_to_gallery("person_001", self.test_features_1)
        self.matcher.add_to_gallery("person_002", self.test_features_2)
        
        # Export gallery
        export_path = self.temp_gallery.name + "_export"
        success = self.matcher.export_gallery(export_path)
        assert success
        assert os.path.exists(export_path)
        
        # Create new matcher and import gallery
        import_matcher = ReIDMatcher(
            device="cpu",
            gallery_path=self.temp_gallery.name + "_import",
            max_gallery_size=100
        )
        
        success = import_matcher.import_gallery(export_path, merge=False)
        assert success
        assert len(import_matcher.gallery) == 2
        assert "person_001" in import_matcher.gallery
        assert "person_002" in import_matcher.gallery
        
        # Clean up
        os.unlink(export_path)
        if os.path.exists(import_matcher.gallery_path):
            os.unlink(import_matcher.gallery_path)
    
    def test_gallery_persistence(self):
        """Test gallery persistence across matcher instances."""
        # Add person to gallery
        self.matcher.add_to_gallery("persistent_person", self.test_features_1)
        
        # Create new matcher with same gallery path
        new_matcher = ReIDMatcher(
            device="cpu",
            gallery_path=self.temp_gallery.name,
            max_gallery_size=100
        )
        
        # Should load existing gallery
        assert len(new_matcher.gallery) == 1
        assert "persistent_person" in new_matcher.gallery


class TestReIDPerformance:
    """Performance tests for re-identification system."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.matcher = ReIDMatcher(device="cpu", max_gallery_size=1000)
        
        # Create large gallery for performance testing
        for i in range(100):
            features = np.random.rand(512).astype(np.float32)
            self.matcher.add_to_gallery(f"person_{i:03d}", features)
    
    def test_matching_performance(self):
        """Test matching performance with large gallery."""
        import time
        
        query_features = np.random.rand(512).astype(np.float32)
        
        # Measure matching time
        start_time = time.time()
        result = self.matcher.match_person(query_features)
        matching_time = time.time() - start_time
        
        # Should complete within reasonable time (< 10ms for 100 persons)
        assert matching_time < 0.01
    
    def test_feature_extraction_performance(self):
        """Test feature extraction performance."""
        import time
        
        person_crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        
        # Measure extraction time
        start_time = time.time()
        features = self.matcher.extract_features(person_crop)
        extraction_time = time.time() - start_time
        
        # Should complete within reasonable time (< 100ms)
        assert extraction_time < 0.1
        assert features.shape == (512,)


class TestReIDLightingConditions:
    """Test re-identification under different lighting conditions."""
    
    def setup_method(self):
        """Set up lighting condition test fixtures."""
        self.matcher = ReIDMatcher(device="cpu")
    
    def _simulate_lighting_change(self, image: np.ndarray, brightness_factor: float) -> np.ndarray:
        """Simulate lighting change by adjusting brightness."""
        adjusted = image.astype(np.float32) * brightness_factor
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def test_reid_under_different_lighting(self):
        """Test re-identification performance under different lighting conditions."""
        # Create base person image
        base_image = np.random.randint(50, 200, (256, 128, 3), dtype=np.uint8)
        base_features = self.matcher.extract_features(base_image)
        
        # Add to gallery
        self.matcher.add_to_gallery("test_person", base_features)
        
        # Test under different lighting conditions
        lighting_factors = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]  # Dark to bright
        successful_matches = 0
        
        for factor in lighting_factors:
            # Simulate lighting change
            modified_image = self._simulate_lighting_change(base_image, factor)
            modified_features = self.matcher.extract_features(modified_image)
            
            # Test matching
            result = self.matcher.match_person(modified_features, threshold=0.3)
            
            if result is not None and result[0] == "test_person":
                successful_matches += 1
        
        # Should successfully match under most lighting conditions
        success_rate = successful_matches / len(lighting_factors)
        assert success_rate >= 0.6  # At least 60% success rate
    
    def test_reid_robustness_to_noise(self):
        """Test re-identification robustness to image noise."""
        # Create base person image
        base_image = np.random.randint(50, 200, (256, 128, 3), dtype=np.uint8)
        base_features = self.matcher.extract_features(base_image)
        
        # Add to gallery
        self.matcher.add_to_gallery("noise_test_person", base_features)
        
        # Test with different noise levels
        noise_levels = [0, 5, 10, 15, 20, 25]  # Standard deviation of Gaussian noise
        successful_matches = 0
        
        for noise_std in noise_levels:
            # Add Gaussian noise
            noise = np.random.normal(0, noise_std, base_image.shape).astype(np.int16)
            noisy_image = np.clip(base_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            noisy_features = self.matcher.extract_features(noisy_image)
            
            # Test matching
            result = self.matcher.match_person(noisy_features, threshold=0.4)
            
            if result is not None and result[0] == "noise_test_person":
                successful_matches += 1
        
        # Should be robust to moderate noise
        success_rate = successful_matches / len(noise_levels)
        assert success_rate >= 0.7  # At least 70% success rate