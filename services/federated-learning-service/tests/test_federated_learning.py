"""
Tests for Federated Learning System
Tests model updates, aggregation, and privacy-preserving features
"""

import pytest
import asyncio
import numpy as np
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from main import FederatedLearningService, ModelUpdate, AggregationRequest
from federated_aggregator import FederatedAggregator, ModelWeights, AggregationResult
from privacy_preserving import PrivacyPreservingTrainer, DifferentialPrivacy

class TestFederatedLearningService:
    """Test federated learning service core functionality"""
    
    @pytest.fixture
    def fl_service(self):
        """Create federated learning service instance"""
        return FederatedLearningService()
    
    @pytest.fixture
    def mock_model_weights(self):
        """Create mock model weights"""
        return {
            "layer1.weight": np.random.randn(64, 32).astype(np.float32),
            "layer1.bias": np.random.randn(64).astype(np.float32),
            "layer2.weight": np.random.randn(10, 64).astype(np.float32),
            "layer2.bias": np.random.randn(10).astype(np.float32)
        }
    
    @pytest.fixture
    def model_update(self, mock_model_weights):
        """Create sample model update"""
        return ModelUpdate(
            edge_node_id="edge-001",
            model_version="v1.0.0",
            weights=mock_model_weights,
            training_samples=1000,
            training_loss=0.25,
            validation_accuracy=0.92,
            privacy_budget_used=0.1,
            metadata={
                "training_duration": 300,
                "local_epochs": 5,
                "batch_size": 32
            }
        )
    
    @pytest.mark.asyncio
    async def test_submit_model_update(self, fl_service, model_update):
        """Test submitting model update from edge node"""
        update_id = await fl_service.submit_model_update(model_update)
        
        assert update_id is not None
        assert update_id in fl_service.pending_updates
        
        stored_update = fl_service.pending_updates[update_id]
        assert stored_update.edge_node_id == model_update.edge_node_id
        assert stored_update.training_samples == model_update.training_samples
        assert stored_update.training_loss == model_update.training_loss
    
    @pytest.mark.asyncio
    async def test_aggregate_models(self, fl_service):
        """Test model aggregation from multiple edge nodes"""
        # Submit multiple model updates
        updates = []
        for i in range(3):
            weights = {
                "layer1.weight": np.random.randn(64, 32).astype(np.float32),
                "layer1.bias": np.random.randn(64).astype(np.float32)
            }
            
            update = ModelUpdate(
                edge_node_id=f"edge-{i:03d}",
                model_version="v1.0.0",
                weights=weights,
                training_samples=1000 + i * 100,
                training_loss=0.3 - i * 0.05,
                validation_accuracy=0.85 + i * 0.02,
                privacy_budget_used=0.1
            )
            
            update_id = await fl_service.submit_model_update(update)
            updates.append(update_id)
        
        # Request aggregation
        aggregation_request = AggregationRequest(
            round_id="round-001",
            model_version="v1.0.0",
            min_participants=2,
            aggregation_method="federated_averaging"
        )
        
        result = await fl_service.aggregate_models(aggregation_request)
        
        assert result is not None
        assert result.round_id == "round-001"
        assert result.participants_count >= 2
        assert result.aggregated_weights is not None
        assert "layer1.weight" in result.aggregated_weights
    
    @pytest.mark.asyncio
    async def test_distribute_global_model(self, fl_service):
        """Test distributing global model to edge nodes"""
        # First create an aggregated model
        weights = {
            "layer1.weight": np.random.randn(64, 32).astype(np.float32),
            "layer1.bias": np.random.randn(64).astype(np.float32)
        }
        
        aggregation_request = AggregationRequest(
            round_id="round-001",
            model_version="v1.0.0",
            min_participants=1,
            aggregation_method="federated_averaging"
        )
        
        # Mock aggregation result
        fl_service.global_models["v1.1.0"] = {
            "weights": weights,
            "version": "v1.1.0",
            "round_id": "round-001",
            "created_at": datetime.now()
        }
        
        # Distribute to edge nodes
        edge_nodes = ["edge-001", "edge-002", "edge-003"]
        distribution_id = await fl_service.distribute_global_model("v1.1.0", edge_nodes)
        
        assert distribution_id is not None
        assert distribution_id in fl_service.model_distributions
        
        distribution = fl_service.model_distributions[distribution_id]
        assert distribution["model_version"] == "v1.1.0"
        assert len(distribution["target_nodes"]) == 3
    
    @pytest.mark.asyncio
    async def test_privacy_budget_tracking(self, fl_service, model_update):
        """Test privacy budget tracking for differential privacy"""
        edge_node_id = model_update.edge_node_id
        
        # Submit multiple updates to track budget
        for i in range(5):
            update = ModelUpdate(
                edge_node_id=edge_node_id,
                model_version="v1.0.0",
                weights=model_update.weights,
                training_samples=1000,
                training_loss=0.25,
                validation_accuracy=0.92,
                privacy_budget_used=0.2  # Each update uses 0.2 budget
            )
            
            await fl_service.submit_model_update(update)
        
        # Check privacy budget
        budget_info = fl_service.get_privacy_budget(edge_node_id)
        
        assert budget_info is not None
        assert budget_info["total_used"] == 1.0  # 5 * 0.2
        assert budget_info["remaining"] <= 0.0  # Should be at or over limit

class TestFederatedAggregator:
    """Test federated aggregation algorithms"""
    
    @pytest.fixture
    def aggregator(self):
        """Create federated aggregator instance"""
        return FederatedAggregator()
    
    @pytest.fixture
    def sample_model_weights(self):
        """Create sample model weights for testing"""
        return [
            ModelWeights(
                edge_node_id="edge-001",
                weights={
                    "layer1": np.array([[1.0, 2.0], [3.0, 4.0]]),
                    "layer2": np.array([0.5, 1.5])
                },
                training_samples=1000,
                quality_score=0.9
            ),
            ModelWeights(
                edge_node_id="edge-002",
                weights={
                    "layer1": np.array([[2.0, 3.0], [4.0, 5.0]]),
                    "layer2": np.array([1.0, 2.0])
                },
                training_samples=800,
                quality_score=0.85
            ),
            ModelWeights(
                edge_node_id="edge-003",
                weights={
                    "layer1": np.array([[1.5, 2.5], [3.5, 4.5]]),
                    "layer2": np.array([0.75, 1.75])
                },
                training_samples=1200,
                quality_score=0.95
            )
        ]
    
    def test_federated_averaging(self, aggregator, sample_model_weights):
        """Test federated averaging aggregation"""
        result = aggregator.federated_averaging(sample_model_weights)
        
        assert isinstance(result, AggregationResult)
        assert result.method == "federated_averaging"
        assert result.participants_count == 3
        
        # Check aggregated weights
        assert "layer1" in result.aggregated_weights
        assert "layer2" in result.aggregated_weights
        
        # Verify weighted average calculation
        expected_layer1 = (
            sample_model_weights[0].weights["layer1"] * 1000 +
            sample_model_weights[1].weights["layer1"] * 800 +
            sample_model_weights[2].weights["layer1"] * 1200
        ) / (1000 + 800 + 1200)
        
        np.testing.assert_array_almost_equal(
            result.aggregated_weights["layer1"],
            expected_layer1,
            decimal=5
        )
    
    def test_quality_weighted_aggregation(self, aggregator, sample_model_weights):
        """Test quality-weighted aggregation"""
        result = aggregator.quality_weighted_aggregation(sample_model_weights)
        
        assert isinstance(result, AggregationResult)
        assert result.method == "quality_weighted"
        assert result.participants_count == 3
        
        # Higher quality models should have more influence
        assert result.aggregated_weights is not None
        assert "layer1" in result.aggregated_weights
    
    def test_secure_aggregation(self, aggregator, sample_model_weights):
        """Test secure aggregation with noise addition"""
        result = aggregator.secure_aggregation(
            sample_model_weights,
            noise_scale=0.1,
            privacy_budget=1.0
        )
        
        assert isinstance(result, AggregationResult)
        assert result.method == "secure_aggregation"
        assert result.privacy_budget_used > 0
        
        # Aggregated weights should be different due to noise
        regular_result = aggregator.federated_averaging(sample_model_weights)
        
        # Should not be exactly equal due to noise
        assert not np.array_equal(
            result.aggregated_weights["layer1"],
            regular_result.aggregated_weights["layer1"]
        )
    
    def test_byzantine_robust_aggregation(self, aggregator):
        """Test Byzantine-robust aggregation with malicious updates"""
        # Create normal updates
        normal_weights = [
            ModelWeights(
                edge_node_id=f"edge-{i:03d}",
                weights={
                    "layer1": np.array([[1.0, 2.0], [3.0, 4.0]]) + np.random.normal(0, 0.1, (2, 2)),
                    "layer2": np.array([0.5, 1.5]) + np.random.normal(0, 0.1, 2)
                },
                training_samples=1000,
                quality_score=0.9
            )
            for i in range(5)
        ]
        
        # Add malicious update
        malicious_weight = ModelWeights(
            edge_node_id="malicious-001",
            weights={
                "layer1": np.array([[100.0, 200.0], [300.0, 400.0]]),  # Extremely large values
                "layer2": np.array([50.0, 150.0])
            },
            training_samples=1000,
            quality_score=0.9
        )
        
        all_weights = normal_weights + [malicious_weight]
        
        result = aggregator.byzantine_robust_aggregation(all_weights)
        
        assert isinstance(result, AggregationResult)
        assert result.method == "byzantine_robust"
        
        # Result should not be heavily influenced by malicious update
        # The aggregated weights should be closer to normal range
        assert np.max(result.aggregated_weights["layer1"]) < 50.0  # Much less than malicious values

class TestPrivacyPreservingTrainer:
    """Test privacy-preserving training mechanisms"""
    
    @pytest.fixture
    def privacy_trainer(self):
        """Create privacy-preserving trainer"""
        return PrivacyPreservingTrainer()
    
    @pytest.fixture
    def sample_gradients(self):
        """Create sample gradients for testing"""
        return {
            "layer1.weight": np.random.randn(64, 32).astype(np.float32),
            "layer1.bias": np.random.randn(64).astype(np.float32),
            "layer2.weight": np.random.randn(10, 64).astype(np.float32),
            "layer2.bias": np.random.randn(10).astype(np.float32)
        }
    
    def test_differential_privacy_noise(self, privacy_trainer, sample_gradients):
        """Test differential privacy noise addition"""
        epsilon = 1.0
        delta = 1e-5
        sensitivity = 1.0
        
        noisy_gradients = privacy_trainer.add_differential_privacy_noise(
            sample_gradients, epsilon, delta, sensitivity
        )
        
        assert len(noisy_gradients) == len(sample_gradients)
        
        # Noisy gradients should be different from original
        for key in sample_gradients:
            assert not np.array_equal(sample_gradients[key], noisy_gradients[key])
            # But should have same shape
            assert sample_gradients[key].shape == noisy_gradients[key].shape
    
    def test_gradient_clipping(self, privacy_trainer, sample_gradients):
        """Test gradient clipping for privacy"""
        clip_norm = 1.0
        
        clipped_gradients = privacy_trainer.clip_gradients(sample_gradients, clip_norm)
        
        # Calculate L2 norm of clipped gradients
        total_norm = 0
        for grad in clipped_gradients.values():
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        # Should be at most clip_norm
        assert total_norm <= clip_norm + 1e-6  # Small tolerance for floating point
    
    def test_secure_aggregation_protocol(self, privacy_trainer):
        """Test secure aggregation protocol"""
        # Simulate multiple clients
        client_updates = []
        for i in range(3):
            update = {
                "layer1": np.random.randn(10, 5).astype(np.float32),
                "layer2": np.random.randn(5).astype(np.float32)
            }
            client_updates.append(update)
        
        # Perform secure aggregation
        aggregated_update = privacy_trainer.secure_aggregate(client_updates)
        
        assert aggregated_update is not None
        assert "layer1" in aggregated_update
        assert "layer2" in aggregated_update
        
        # Verify aggregation (should be sum of all updates)
        expected_layer1 = sum(update["layer1"] for update in client_updates)
        np.testing.assert_array_almost_equal(
            aggregated_update["layer1"],
            expected_layer1,
            decimal=5
        )
    
    def test_privacy_budget_calculation(self, privacy_trainer):
        """Test privacy budget calculation"""
        epsilon = 1.0
        delta = 1e-5
        num_rounds = 10
        
        budget_per_round = privacy_trainer.calculate_privacy_budget_per_round(
            total_epsilon=epsilon,
            total_delta=delta,
            num_rounds=num_rounds
        )
        
        assert budget_per_round["epsilon"] <= epsilon / num_rounds
        assert budget_per_round["delta"] <= delta / num_rounds
        
        # Total budget should not exceed limits
        total_used_epsilon = budget_per_round["epsilon"] * num_rounds
        total_used_delta = budget_per_round["delta"] * num_rounds
        
        assert total_used_epsilon <= epsilon
        assert total_used_delta <= delta

class TestIntegrationScenarios:
    """Test end-to-end federated learning scenarios"""
    
    @pytest.fixture
    def fl_service(self):
        """Create federated learning service"""
        return FederatedLearningService()
    
    @pytest.mark.asyncio
    async def test_complete_federated_round(self, fl_service):
        """Test complete federated learning round"""
        # 1. Multiple edge nodes submit updates
        edge_nodes = ["edge-001", "edge-002", "edge-003"]
        update_ids = []
        
        for i, node_id in enumerate(edge_nodes):
            weights = {
                "layer1": np.random.randn(32, 16).astype(np.float32),
                "layer2": np.random.randn(10, 32).astype(np.float32)
            }
            
            update = ModelUpdate(
                edge_node_id=node_id,
                model_version="v1.0.0",
                weights=weights,
                training_samples=1000 + i * 100,
                training_loss=0.3 - i * 0.02,
                validation_accuracy=0.85 + i * 0.03,
                privacy_budget_used=0.1
            )
            
            update_id = await fl_service.submit_model_update(update)
            update_ids.append(update_id)
        
        # 2. Aggregate models
        aggregation_request = AggregationRequest(
            round_id="round-001",
            model_version="v1.0.0",
            min_participants=3,
            aggregation_method="federated_averaging"
        )
        
        aggregation_result = await fl_service.aggregate_models(aggregation_request)
        
        assert aggregation_result is not None
        assert aggregation_result.participants_count == 3
        
        # 3. Create new global model version
        new_version = "v1.1.0"
        fl_service.global_models[new_version] = {
            "weights": aggregation_result.aggregated_weights,
            "version": new_version,
            "round_id": "round-001",
            "created_at": datetime.now(),
            "performance_metrics": {
                "avg_loss": aggregation_result.avg_loss,
                "avg_accuracy": aggregation_result.avg_accuracy
            }
        }
        
        # 4. Distribute new model
        distribution_id = await fl_service.distribute_global_model(new_version, edge_nodes)
        
        assert distribution_id is not None
        
        # 5. Verify distribution
        distribution = fl_service.model_distributions[distribution_id]
        assert distribution["model_version"] == new_version
        assert len(distribution["target_nodes"]) == 3
        assert distribution["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_privacy_preserving_round(self, fl_service):
        """Test federated learning round with privacy preservation"""
        # Submit updates with privacy budget tracking
        edge_nodes = ["edge-001", "edge-002"]
        
        for node_id in edge_nodes:
            weights = {
                "layer1": np.random.randn(16, 8).astype(np.float32),
                "layer2": np.random.randn(5, 16).astype(np.float32)
            }
            
            update = ModelUpdate(
                edge_node_id=node_id,
                model_version="v1.0.0",
                weights=weights,
                training_samples=800,
                training_loss=0.25,
                validation_accuracy=0.90,
                privacy_budget_used=0.5  # High privacy budget usage
            )
            
            await fl_service.submit_model_update(update)
        
        # Aggregate with secure aggregation
        aggregation_request = AggregationRequest(
            round_id="privacy-round-001",
            model_version="v1.0.0",
            min_participants=2,
            aggregation_method="secure_aggregation",
            privacy_budget=1.0
        )
        
        result = await fl_service.aggregate_models(aggregation_request)
        
        assert result is not None
        assert result.method == "secure_aggregation"
        assert result.privacy_budget_used > 0
        
        # Check privacy budgets are updated
        for node_id in edge_nodes:
            budget_info = fl_service.get_privacy_budget(node_id)
            assert budget_info["total_used"] >= 0.5
    
    @pytest.mark.asyncio
    async def test_model_performance_tracking(self, fl_service):
        """Test tracking model performance across rounds"""
        rounds_data = []
        
        # Simulate multiple rounds with improving performance
        for round_num in range(3):
            edge_nodes = ["edge-001", "edge-002", "edge-003"]
            
            # Submit updates with improving metrics
            for i, node_id in enumerate(edge_nodes):
                weights = {
                    "layer1": np.random.randn(16, 8).astype(np.float32)
                }
                
                # Simulate improving performance
                base_loss = 0.5 - round_num * 0.1
                base_accuracy = 0.8 + round_num * 0.05
                
                update = ModelUpdate(
                    edge_node_id=node_id,
                    model_version=f"v1.{round_num}.0",
                    weights=weights,
                    training_samples=1000,
                    training_loss=base_loss + np.random.normal(0, 0.02),
                    validation_accuracy=base_accuracy + np.random.normal(0, 0.01),
                    privacy_budget_used=0.1
                )
                
                await fl_service.submit_model_update(update)
            
            # Aggregate
            aggregation_request = AggregationRequest(
                round_id=f"round-{round_num:03d}",
                model_version=f"v1.{round_num}.0",
                min_participants=3,
                aggregation_method="federated_averaging"
            )
            
            result = await fl_service.aggregate_models(aggregation_request)
            rounds_data.append(result)
        
        # Verify performance improvement trend
        losses = [round_data.avg_loss for round_data in rounds_data]
        accuracies = [round_data.avg_accuracy for round_data in rounds_data]
        
        # Loss should generally decrease
        assert losses[2] < losses[0]  # Final loss better than initial
        
        # Accuracy should generally increase
        assert accuracies[2] > accuracies[0]  # Final accuracy better than initial

if __name__ == "__main__":
    pytest.main([__file__, "-v"])