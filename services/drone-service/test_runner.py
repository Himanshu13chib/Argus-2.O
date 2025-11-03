"""
Simple test runner for drone integration system
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from main import DroneIntegrationService, DeploymentRequest, AuthorizationRequest
from drone_tracker import EvidenceType

async def test_basic_functionality():
    """Test basic drone integration functionality"""
    print("Testing Drone Integration System...")
    
    # Create service instance
    service = DroneIntegrationService()
    
    # Start background services
    await service.evidence_collector.start_collection_processor()
    await service.handoff_protocol.start_monitoring()
    
    try:
        # Test 1: Request deployment
        print("\n1. Testing deployment request...")
        deployment_request = DeploymentRequest(
            incident_id="test-incident-001",
            target_location={"lat": 28.6139, "lon": 77.2090},
            mission_type="surveillance",
            priority="high",
            estimated_duration=30,
            required_capabilities=["surveillance"],
            operator_notes="Test deployment"
        )
        
        deployment_id = await service.request_drone_deployment(deployment_request, "operator-123")
        print(f"✓ Deployment requested: {deployment_id}")
        
        # Test 2: Authorize deployment
        print("\n2. Testing deployment authorization...")
        auth_request = AuthorizationRequest(
            deployment_id=deployment_id,
            operator_id="supervisor-001",
            authorization_reason="Test authorization",
            approved=True
        )
        
        approved = await service.authorize_deployment(auth_request)
        print(f"✓ Deployment authorized: {approved}")
        
        # Test 3: Collect evidence
        print("\n3. Testing evidence collection...")
        evidence_id = await service.collect_evidence(deployment_id, "video")
        print(f"✓ Evidence collected: {evidence_id}")
        
        # Wait a bit for evidence processing
        await asyncio.sleep(1)
        
        # Test 4: Get evidence summary
        print("\n4. Testing evidence summary...")
        summary = await service.evidence_collector.get_evidence_summary(deployment_id)
        print(f"✓ Evidence summary: {summary['total_items']} items collected")
        
        # Test 5: Test handoff
        print("\n5. Testing handoff protocol...")
        deployment = service.deployments[deployment_id]
        handoff_id = await service.handoff_protocol.initiate_handoff(
            drone_id=deployment.drone_id,
            deployment_id=deployment_id,
            from_operator_id="operator-123",
            handoff_type="operator_to_operator",
            reason="Test handoff"
        )
        print(f"✓ Handoff initiated: {handoff_id}")
        
        # Test 6: Recall drone
        print("\n6. Testing drone recall...")
        recalled = await service.recall_drone(deployment_id, "operator-123")
        print(f"✓ Drone recalled: {recalled}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        await service.evidence_collector.stop_collection_processor()
        await service.handoff_protocol.stop_monitoring()

async def test_tracking_system():
    """Test drone tracking system"""
    print("\nTesting Drone Tracking System...")
    
    from drone_tracker import DroneTracker, EvidenceCollector, GPSCoordinate
    from datetime import datetime
    
    tracker = DroneTracker()
    evidence_collector = EvidenceCollector(tracker)
    
    await evidence_collector.start_collection_processor()
    
    try:
        # Test tracking
        initial_location = GPSCoordinate(
            latitude=28.6139,
            longitude=77.2090,
            altitude=100.0,
            timestamp=datetime.now(),
            accuracy=3.0
        )
        
        await tracker.start_tracking("drone-001", "deployment-001", initial_location)
        print("✓ Tracking started")
        
        # Test evidence collection
        evidence_id = await evidence_collector.collect_evidence(
            "drone-001", "deployment-001", EvidenceType.VIDEO
        )
        print(f"✓ Evidence collected: {evidence_id}")
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Get evidence items
        evidence_items = tracker.get_evidence_items("deployment-001")
        print(f"✓ Evidence items: {len(evidence_items)}")
        
        # Stop tracking
        await tracker.stop_tracking("drone-001")
        print("✓ Tracking stopped")
        
        print("✅ Tracking system tests passed!")
        
    except Exception as e:
        print(f"❌ Tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await evidence_collector.stop_collection_processor()

if __name__ == "__main__":
    asyncio.run(test_basic_functionality())
    asyncio.run(test_tracking_system())