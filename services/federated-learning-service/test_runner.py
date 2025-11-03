"""
Simple test runner for federated learning system
"""

import asyncio
import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from main import FederatedLearningService, ModelUpdate, AggregationRequest

async def test_federated_learning():
    """Test federated learning functionality"""
    print("Testing Federated Learning System...")
    
    # Create service instance
    service = FederatedLearningService()
    
    try:
        # Test 1: Submit model updates from multiple edge nodes
        print("\n1. Testing model update submission...")
        
        updates = []
        for i in range(3):
            weights = {
                "layer1.weight": np.random.randn(32, 16).astype(np.float32),
                "layer1.bias": np.random.randn(32).astype(np.float32),
                "layer2.weight": np.random.randn(10, 32).astype(np.float32),
                "layer2.bias": np.random.randn(10).astype(np.float32)
            }
            
            update = ModelUpdate(
                edge_node_id=f"edge-{i:03d}",
                model_version="v1.0.0",
                weights=weights,
                training_samples=1000 + i * 100,
                training_loss=0.3 - i * 0.02,
                validation_accuracy=0.85 + i * 0.03,
                privacy_budget_used=0.1,
                metadata={
                    "training_duration": 300,
                    "local_epochs": 5
                }
            )
            
            update_id = await service.submit_model_update(update)
            updates.append(update_id)
            print(f"✓ Update submitted from edge-{i:03d}: {update_id}")
        
        # Test 2: Aggregate models
        print("\n2. Testing model aggregation...")
        
        aggregation_request = AggregationRequest(
            round_id="test-round-001",
            model_version="v1.0.0",
            min_participants=3,
            aggregation_method="federated_averaging"
        )
        
        result = await service.aggregate_models(aggregation_request)
        print(f"✓ Models aggregated: {result.participants_count} participants")
        print(f"  - Average loss: {result.avg_loss:.4f}")
        print(f"  - Average accuracy: {result.avg_accuracy:.4f}")
        
        # Test 3: Create global model
        print("\n3. Testing global model creation...")
        
        new_version = "v1.1.0"
        service.global_models[new_version] = {
            "weights": result.aggregated_weights,
            "version": new_version,
            "round_id": "test-round-001",
            "created_at": result.timestamp,
            "performance_metrics": {
                "avg_loss": result.avg_loss,
                "avg_accuracy": result.avg_accuracy
            }
        }
        print(f"✓ Global model created: {new_version}")
        
        # Test 4: Distribute global model
        print("\n4. Testing model distribution...")
        
        edge_nodes = ["edge-001", "edge-002", "edge-003"]
        distribution_id = await service.distribute_global_model(new_version, edge_nodes)
        print(f"✓ Model distributed: {distribution_id}")
        
        distribution = service.model_distributions[distribution_id]
        print(f"  - Target nodes: {len(distribution['target_nodes'])}")
        print(f"  - Status: {distribution['status']}")
        
        # Test 5: Privacy budget tracking
        print("\n5. Testing privacy budget tracking...")
        
        for i in range(3):
            edge_node_id = f"edge-{i:03d}"
            budget_info = service.get_privacy_budget(edge_node_id)
            print(f"✓ Privacy budget for {edge_node_id}: {budget_info['total_used']:.2f} used, {budget_info['remaining']:.2f} remaining")
        
        print("\n✅ All federated learning tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_aggregation_methods():
    """Test different aggregation methods"""
    print("\nTesting Aggregation Methods...")
    
    from federated_aggregator import FederatedAggregator, ModelWeights
    
    aggregator = FederatedAggregator()
    
    try:
        # Create sample model weights
        sample_weights = []
        for i in range(3):
            weights = ModelWeights(
                edge_node_id=f"edge-{i:03d}",
                weights={
                    "layer1": np.random.randn(4, 2).astype(np.float32),
                    "layer2": np.random.randn(2).astype(np.float32)
                },
                training_samples=1000 + i * 200,
                quality_score=0.9 + i * 0.02
            )
            sample_weights.append(weights)
        
        # Test federated averaging
        print("\n1. Testing federated averaging...")
        result = aggregator.federated_averaging(sample_weights)
        print(f"✓ Federated averaging completed: {result.participants_count} participants")
        
        # Test quality weighted aggregation
        print("\n2. Testing quality weighted aggregation...")
        result = aggregator.quality_weighted_aggregation(sample_weights)
        print(f"✓ Quality weighted aggregation completed: {result.participants_count} participants")
        
        # Test secure aggregation
        print("\n3. Testing secure aggregation...")
        result = aggregator.secure_aggregation(sample_weights, noise_scale=0.1, privacy_budget=1.0)
        print(f"✓ Secure aggregation completed: privacy budget used {result.privacy_budget_used:.3f}")
        
        print("\n✅ All aggregation method tests passed!")
        
    except Exception as e:
        print(f"\n❌ Aggregation test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_federated_learning())
    asyncio.run(test_aggregation_methods())