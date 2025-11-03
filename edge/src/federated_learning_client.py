"""
Federated Learning Client for Edge Nodes
Handles local model training and secure updates to central coordinator
"""

import asyncio
import logging
import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import requests
from dataclasses import dataclass

from shared.models.detection import ModelUpdate, FederatedLearningConfig
from detection_pipeline import DetectionPipeline

logger = logging.getLogger(__name__)

@dataclass
class LocalTrainingData:
    """Local training data for federated learning"""
    images: List[np.ndarray]
    labels: List[Dict]
    timestamps: List[datetime]
    quality_scores: List[float]

class FederatedLearningClient:
    """Edge node client for federated learning"""
    
    def __init__(self, edge_node_id: str, coordinator_url: str):
        self.edge_node_id = edge_node_id
        self.coordinator_url = coordinator_url
        self.encryption_key = self._generate_encryption_key()
        self.local_model = None
        self.training_data: LocalTrainingData = LocalTrainingData([], [], [], [])
        self.config = self._load_config()
        
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key matching coordinator"""
        password = os.getenv('FL_ENCRYPTION_PASSWORD', 'default_password').encode()
        salt = os.getenv('FL_SALT', 'default_salt').encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password))
    
    def _load_config(self) -> Dict:
        """Load federated learning configuration"""
        return {
            "min_training_samples": int(os.getenv('FL_MIN_SAMPLES', '100')),
            "training_epochs": int(os.getenv('FL_EPOCHS', '5')),
            "learning_rate": float(os.getenv('FL_LEARNING_RATE', '0.001')),
            "privacy_budget": float(os.getenv('FL_PRIVACY_BUDGET', '1.0')),
            "update_frequency_hours": int(os.getenv('FL_UPDATE_FREQ', '24'))
        }
    
    async def collect_training_data(self, detection_result: Dict, image: np.ndarray):
        """Collect local training data from detections"""
        # Only collect high-quality detections for training
        if detection_result.get('confidence', 0) > 0.8:
            # Add differential privacy noise to protect individual data points
            noisy_image = self._add_differential_privacy_noise(image)
            
            self.training_data.images.append(noisy_image)
            self.training_data.labels.append(detection_result)
            self.training_data.timestamps.append(datetime.now())
            self.training_data.quality_scores.append(detection_result.get('confidence', 0))
            
            # Limit training data size to prevent memory issues
            max_samples = self.config['min_training_samples'] * 2
            if len(self.training_data.images) > max_samples:
                # Remove oldest samples
                self.training_data.images = self.training_data.images[-max_samples:]
                self.training_data.labels = self.training_data.labels[-max_samples:]
                self.training_data.timestamps = self.training_data.timestamps[-max_samples:]
                self.training_data.quality_scores = self.training_data.quality_scores[-max_samples:]
    
    def _add_differential_privacy_noise(self, image: np.ndarray) -> np.ndarray:
        """Add differential privacy noise to protect individual privacy"""
        # Add Gaussian noise calibrated to privacy budget
        noise_scale = 1.0 / self.config['privacy_budget']
        noise = np.random.normal(0, noise_scale, image.shape)
        
        # Clip noise to prevent extreme values
        noise = np.clip(noise, -0.1, 0.1)
        
        # Add noise and ensure valid pixel range
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    async def train_local_model(self) -> Optional[ModelUpdate]:
        """Train local model with collected data"""
        if len(self.training_data.images) < self.config['min_training_samples']:
            logger.info(f"Insufficient training data: {len(self.training_data.images)} < {self.config['min_training_samples']}")
            return None
        
        logger.info(f"Starting local training with {len(self.training_data.images)} samples")
        
        try:
            # Create a lightweight model for federated learning
            if self.local_model is None:
                self.local_model = self._create_lightweight_model()
            
            # Prepare training data
            X, y = self._prepare_training_data()
            
            # Train model with differential privacy
            model_weights = await self._train_with_privacy(X, y)
            
            # Create model update
            update = ModelUpdate(
                model_name="person_detection",
                version=f"edge_{self.edge_node_id}_{datetime.now().isoformat()}",
                weights=model_weights.tolist(),
                timestamp=datetime.now(),
                edge_node_id=self.edge_node_id,
                training_samples=len(self.training_data.images),
                privacy_budget_used=self.config['privacy_budget']
            )
            
            logger.info("Local model training completed")
            return update
            
        except Exception as e:
            logger.error(f"Error in local training: {e}")
            return None
    
    def _create_lightweight_model(self) -> nn.Module:
        """Create lightweight model for federated learning"""
        class LightweightDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((8, 8))
                self.fc1 = nn.Linear(64 * 8 * 8, 128)
                self.fc2 = nn.Linear(128, 2)  # person/no-person
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.5)
            
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        return LightweightDetector()
    
    def _prepare_training_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare training data for PyTorch model"""
        # Convert images to tensors
        images = []
        labels = []
        
        for img, label in zip(self.training_data.images, self.training_data.labels):
            # Resize and normalize image
            img_resized = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            img_resized = torch.nn.functional.interpolate(
                img_resized.unsqueeze(0), size=(64, 64), mode='bilinear'
            ).squeeze(0)
            
            images.append(img_resized)
            
            # Convert detection to binary classification
            has_person = 1 if len(label.get('detections', [])) > 0 else 0
            labels.append(has_person)
        
        X = torch.stack(images)
        y = torch.tensor(labels, dtype=torch.long)
        
        return X, y
    
    async def _train_with_privacy(self, X: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """Train model with differential privacy guarantees"""
        optimizer = torch.optim.Adam(self.local_model.parameters(), 
                                   lr=self.config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.local_model.train()
        for epoch in range(self.config['training_epochs']):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.local_model(X)
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Add noise to gradients for differential privacy
            self._add_gradient_noise()
            
            optimizer.step()
            
            if epoch % 2 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Extract model weights
        weights = []
        for param in self.local_model.parameters():
            weights.extend(param.data.flatten().numpy())
        
        return np.array(weights)
    
    def _add_gradient_noise(self):
        """Add noise to gradients for differential privacy"""
        noise_scale = 1.0 / (self.config['privacy_budget'] * len(self.training_data.images))
        
        for param in self.local_model.parameters():
            if param.grad is not None:
                noise = torch.normal(0, noise_scale, param.grad.shape)
                param.grad.add_(noise)
    
    async def submit_model_update(self, model_update: ModelUpdate, round_id: str) -> bool:
        """Submit encrypted model update to coordinator"""
        try:
            # Encrypt model update
            encrypted_update = self._encrypt_model_update(model_update)
            
            # Submit to coordinator
            response = requests.post(
                f"{self.coordinator_url}/federated/submit-update/{round_id}",
                params={"edge_node_id": self.edge_node_id},
                data=encrypted_update,
                headers={"Content-Type": "application/octet-stream"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Model update submitted successfully: {result}")
                return result.get('success', False)
            else:
                logger.error(f"Failed to submit update: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting model update: {e}")
            return False
    
    def _encrypt_model_update(self, model_update: ModelUpdate) -> bytes:
        """Encrypt model update for secure transmission"""
        fernet = Fernet(self.encryption_key)
        update_json = model_update.json().encode()
        return fernet.encrypt(update_json)
    
    async def download_global_model(self, model_name: str) -> Optional[ModelUpdate]:
        """Download latest global model from coordinator"""
        try:
            response = requests.get(
                f"{self.coordinator_url}/federated/latest-model/{model_name}",
                timeout=30
            )
            
            if response.status_code == 200:
                model_data = response.json()
                return ModelUpdate(**model_data)
            else:
                logger.warning(f"No global model available: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading global model: {e}")
            return None
    
    async def update_local_model(self, global_model: ModelUpdate):
        """Update local model with global model weights"""
        try:
            if self.local_model is None:
                self.local_model = self._create_lightweight_model()
            
            # Load global weights into local model
            weights = np.array(global_model.weights)
            param_idx = 0
            
            for param in self.local_model.parameters():
                param_size = param.numel()
                param_weights = weights[param_idx:param_idx + param_size]
                param.data = torch.from_numpy(param_weights.reshape(param.shape)).float()
                param_idx += param_size
            
            logger.info(f"Updated local model with global weights from {global_model.version}")
            
        except Exception as e:
            logger.error(f"Error updating local model: {e}")
    
    async def run_federated_learning_cycle(self):
        """Run complete federated learning cycle"""
        try:
            # Check if we have enough data for training
            if len(self.training_data.images) < self.config['min_training_samples']:
                logger.info("Insufficient data for federated learning cycle")
                return
            
            # Train local model
            model_update = await self.train_local_model()
            if not model_update:
                logger.warning("Local training failed")
                return
            
            # Check for active federated rounds
            status_response = requests.get(f"{self.coordinator_url}/federated/status")
            if status_response.status_code == 200:
                status = status_response.json()
                if status['active_rounds'] > 0:
                    # Find active round ID (simplified - in production, would be more sophisticated)
                    round_id = f"person_detection_{datetime.now().strftime('%Y-%m-%d')}"
                    
                    # Submit update
                    success = await self.submit_model_update(model_update, round_id)
                    if success:
                        logger.info("Successfully participated in federated learning round")
                        
                        # Download updated global model
                        await asyncio.sleep(60)  # Wait for aggregation
                        global_model = await self.download_global_model("person_detection")
                        if global_model:
                            await self.update_local_model(global_model)
            
        except Exception as e:
            logger.error(f"Error in federated learning cycle: {e}")
    
    def clear_training_data(self):
        """Clear local training data after successful update"""
        self.training_data = LocalTrainingData([], [], [], [])
        logger.info("Cleared local training data")

# Global federated learning client instance
fl_client = None

def initialize_federated_learning(edge_node_id: str, coordinator_url: str) -> FederatedLearningClient:
    """Initialize federated learning client"""
    global fl_client
    fl_client = FederatedLearningClient(edge_node_id, coordinator_url)
    logger.info(f"Initialized federated learning client for edge node: {edge_node_id}")
    return fl_client

def get_federated_learning_client() -> Optional[FederatedLearningClient]:
    """Get global federated learning client instance"""
    return fl_client