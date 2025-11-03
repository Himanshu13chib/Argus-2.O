"""
Federated Learning Service for Project Argus
Coordinates model updates across edge nodes while preserving privacy
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer
import uvicorn
from pydantic import BaseModel
import hashlib
import json
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from shared.interfaces.security import SecurityManager
from shared.models.detection import ModelUpdate, FederatedLearningConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Federated Learning Service", version="1.0.0")
security = HTTPBearer()

class FederatedLearningCoordinator:
    """Coordinates federated learning across edge nodes"""
    
    def __init__(self):
        self.active_rounds: Dict[str, FederatedRound] = {}
        self.model_registry: Dict[str, ModelMetadata] = {}
        self.encryption_key = self._generate_encryption_key()
        
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for secure model updates"""
        password = os.getenv('FL_ENCRYPTION_PASSWORD', 'default_password').encode()
        salt = os.getenv('FL_SALT', 'default_salt').encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password))
    
    async def start_federated_round(self, model_name: str, config: FederatedLearningConfig) -> str:
        """Start a new federated learning round"""
        round_id = f"{model_name}_{datetime.now().isoformat()}"
        
        federated_round = FederatedRound(
            round_id=round_id,
            model_name=model_name,
            config=config,
            start_time=datetime.now()
        )
        
        self.active_rounds[round_id] = federated_round
        logger.info(f"Started federated learning round: {round_id}")
        
        return round_id
    
    async def submit_model_update(self, round_id: str, edge_node_id: str, 
                                encrypted_update: bytes) -> bool:
        """Receive encrypted model update from edge node"""
        if round_id not in self.active_rounds:
            raise HTTPException(status_code=404, detail="Federated round not found")
        
        federated_round = self.active_rounds[round_id]
        
        # Decrypt and validate update
        try:
            decrypted_update = self._decrypt_model_update(encrypted_update)
            update = ModelUpdate.parse_raw(decrypted_update)
            
            # Validate update integrity
            if not self._validate_update_integrity(update):
                logger.warning(f"Invalid update from edge node {edge_node_id}")
                return False
            
            federated_round.add_update(edge_node_id, update)
            logger.info(f"Received update from edge node {edge_node_id} for round {round_id}")
            
            # Check if we have enough updates to aggregate
            if len(federated_round.updates) >= federated_round.config.min_participants:
                await self._aggregate_and_distribute(federated_round)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing update from {edge_node_id}: {e}")
            return False
    
    def _decrypt_model_update(self, encrypted_update: bytes) -> bytes:
        """Decrypt model update using symmetric encryption"""
        fernet = Fernet(self.encryption_key)
        return fernet.decrypt(encrypted_update)
    
    def _validate_update_integrity(self, update: ModelUpdate) -> bool:
        """Validate model update integrity and authenticity"""
        # Check update size and format
        if not update.weights or len(update.weights) == 0:
            return False
        
        # Verify digital signature if present
        if update.signature:
            # Implement signature verification logic
            pass
        
        # Check for reasonable weight values (no extreme outliers)
        weights_array = np.array(update.weights)
        if np.any(np.abs(weights_array) > 100):  # Reasonable threshold
            return False
        
        return True
    
    async def _aggregate_and_distribute(self, federated_round: FederatedRound):
        """Aggregate model updates and distribute to edge nodes"""
        logger.info(f"Aggregating {len(federated_round.updates)} updates for round {federated_round.round_id}")
        
        # Federated averaging
        aggregated_weights = self._federated_averaging(federated_round.updates)
        
        # Create new global model
        global_model = ModelUpdate(
            model_name=federated_round.model_name,
            version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            weights=aggregated_weights.tolist(),
            timestamp=datetime.now(),
            round_id=federated_round.round_id
        )
        
        # Store in model registry
        self.model_registry[federated_round.model_name] = ModelMetadata(
            model_name=federated_round.model_name,
            latest_version=global_model.version,
            last_updated=datetime.now(),
            performance_metrics={}
        )
        
        # Mark round as complete
        federated_round.status = "completed"
        federated_round.global_model = global_model
        
        logger.info(f"Completed federated learning round: {federated_round.round_id}")
    
    def _federated_averaging(self, updates: Dict[str, ModelUpdate]) -> np.ndarray:
        """Perform federated averaging of model updates"""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # Simple federated averaging (can be enhanced with weighted averaging)
        all_weights = []
        for update in updates.values():
            all_weights.append(np.array(update.weights))
        
        # Average all weight updates
        aggregated = np.mean(all_weights, axis=0)
        return aggregated
    
    async def get_latest_model(self, model_name: str) -> Optional[ModelUpdate]:
        """Get latest global model for edge nodes"""
        if model_name not in self.model_registry:
            return None
        
        # Find the latest completed round for this model
        for round_id, federated_round in self.active_rounds.items():
            if (federated_round.model_name == model_name and 
                federated_round.status == "completed" and 
                federated_round.global_model):
                return federated_round.global_model
        
        return None

class FederatedRound:
    """Represents a single federated learning round"""
    
    def __init__(self, round_id: str, model_name: str, config: FederatedLearningConfig, start_time: datetime):
        self.round_id = round_id
        self.model_name = model_name
        self.config = config
        self.start_time = start_time
        self.updates: Dict[str, ModelUpdate] = {}
        self.status = "active"
        self.global_model: Optional[ModelUpdate] = None
    
    def add_update(self, edge_node_id: str, update: ModelUpdate):
        """Add model update from edge node"""
        self.updates[edge_node_id] = update

class ModelMetadata(BaseModel):
    """Metadata for registered models"""
    model_name: str
    latest_version: str
    last_updated: datetime
    performance_metrics: Dict[str, float]

# Global coordinator instance
coordinator = FederatedLearningCoordinator()

@app.post("/federated/start-round")
async def start_federated_round(config: FederatedLearningConfig):
    """Start a new federated learning round"""
    try:
        round_id = await coordinator.start_federated_round(config.model_name, config)
        return {"round_id": round_id, "status": "started"}
    except Exception as e:
        logger.error(f"Error starting federated round: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/federated/submit-update/{round_id}")
async def submit_model_update(round_id: str, edge_node_id: str, encrypted_update: bytes):
    """Submit encrypted model update from edge node"""
    try:
        success = await coordinator.submit_model_update(round_id, edge_node_id, encrypted_update)
        return {"success": success}
    except Exception as e:
        logger.error(f"Error submitting update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/federated/latest-model/{model_name}")
async def get_latest_model(model_name: str):
    """Get latest global model for edge nodes"""
    try:
        model = await coordinator.get_latest_model(model_name)
        if model:
            return model.dict()
        else:
            raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        logger.error(f"Error getting latest model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/federated/status")
async def get_federated_status():
    """Get status of all federated learning rounds"""
    return {
        "active_rounds": len([r for r in coordinator.active_rounds.values() if r.status == "active"]),
        "completed_rounds": len([r for r in coordinator.active_rounds.values() if r.status == "completed"]),
        "registered_models": list(coordinator.model_registry.keys())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)