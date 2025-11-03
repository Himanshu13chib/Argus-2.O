"""
Tests for Drone Integration System
Tests drone deployment workflows and authorization procedures
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from main import DroneIntegrationService, DeploymentRequest, AuthorizationRequest
from drone_tracker import DroneTracker, EvidenceCollector, EvidenceType, GPSCoordinate
from handoff_protocols import DroneHandoffProtocol, HandoffType, AuthorizationLevel

class TestDroneIntegrationService:
    """Test drone integration service core functionality"""
    
    @pytest.fixture
    def drone_service(self):
        """Create drone service instance for testing"""
        return DroneIntegrationService()
    
    @pytest.fixture
    def deployment_request(self):
        """Create sample deployment request"""
        return DeploymentRequest(
            incident_id="incident-123",
            target_location={"lat": 28.6139, "lon": 77.2090},
            mission_type="surveillance",
            priority="high",
            estimated_duration=30,
            required_capabilities=["surveillance", "thermal_imaging"],
            operator_notes="Border crossing detected"
        )
    
    @pytest.mark.asyncio
    async def test_request_drone_deployment(self, drone_service, deployment_request):
        """Test drone deployment request creation"""
        operator_id = "operator-123"
        
        deployment_id = await drone_service.request_drone_deployment(
            deployment_request, operator_id
        )
        
        assert deployment_id is not None
        assert deployment_id in drone_service.deployments
        
        deployment = drone_service.deployments[deployment_id]
        assert deployment.incident_id == deployment_request.incident_id
        assert deployment.operator_id == operator_id
        assert deployment.status.value == "pending_authorization"
    
    @pytest.mark.asyncio
    async def test_authorize_deployment_approved(self, drone_service, deployment_request):
        """Test deployment authorization approval"""
        operator_id = "operator-123"
        deployment_id = await drone_service.request_drone_deployment(
            deployment_request, operator_id
        )
        
        auth_request = AuthorizationRequest(
            deployment_id=deployment_id,
            operator_id="supervisor-001",
            authorization_reason="Valid security threat",
            approved=True
        )
        
        result = await drone_service.authorize_deployment(auth_request)
        
        assert result is True
        deployment = drone_service.deployments[deployment_id]
        assert deployment.status.value == "active"
        assert deployment.supervisor_id == "supervisor-001"
        assert deployment.authorization_timestamp is not None
    
    @pytest.mark.asyncio
    async def test_authorize_deployment_denied(self, drone_service, deployment_request):
        """Test deployment authorization denial"""
        operator_id = "operator-123"
        deployment_id = await drone_service.request_drone_deployment(
            deployment_request, operator_id
        )
        
        auth_request = AuthorizationRequest(
            deployment_id=deployment_id,
            operator_id="supervisor-001",
            authorization_reason="Insufficient threat level",
            approved=False
        )
        
        result = await drone_service.authorize_deployment(auth_request)
        
        assert result is False
        deployment = drone_service.deployments[deployment_id]
        assert deployment.status.value == "cancelled"
    
    @pytest.mark.asyncio
    async def test_collect_evidence_during_mission(self, drone_service, deployment_request):
        """Test evidence collection during active mission"""
        operator_id = "operator-123"
        deployment_id = await drone_service.request_drone_deployment(
            deployment_request, operator_id
        )
        
        # Authorize deployment
        auth_request = AuthorizationRequest(
            deployment_id=deployment_id,
            operator_id="supervisor-001",
            authorization_reason="Approved",
            approved=True
        )
        await drone_service.authorize_deployment(auth_request)
        
        # Collect evidence
        evidence_id = await drone_service.collect_evidence(deployment_id, "video")
        
        assert evidence_id is not None
        deployment = drone_service.deployments[deployment_id]
        assert evidence_id in deployment.evidence_collected
    
    @pytest.mark.asyncio
    async def test_recall_drone(self, drone_service, deployment_request):
        """Test drone recall and mission completion"""
        operator_id = "operator-123"
        deployment_id = await drone_service.request_drone_deployment(
            deployment_request, operator_id
        )
        
        # Authorize and deploy
        auth_request = AuthorizationRequest(
            deployment_id=deployment_id,
            operator_id="supervisor-001",
            authorization_reason="Approved",
            approved=True
        )
        await drone_service.authorize_deployment(auth_request)
        
        # Recall drone
        result = await drone_service.recall_drone(deployment_id, operator_id)
        
        assert result is True
        deployment = drone_service.deployments[deployment_id]
        assert deployment.status.value == "completed"
        assert deployment.completion_timestamp is not None

class TestDroneTracker:
    """Test drone tracking and flight path management"""
    
    @pytest.fixture
    def tracker(self):
        """Create drone tracker instance"""
        return DroneTracker()
    
    @pytest.fixture
    def initial_location(self):
        """Create initial GPS location"""
        return GPSCoordinate(
            latitude=28.6139,
            longitude=77.2090,
            altitude=100.0,
            timestamp=datetime.now(),
            accuracy=3.0
        )
    
    @pytest.mark.asyncio
    async def test_start_tracking(self, tracker, initial_location):
        """Test starting drone tracking"""
        drone_id = "drone-001"
        deployment_id = "deployment-123"
        
        await tracker.start_tracking(drone_id, deployment_id, initial_location)
        
        assert drone_id in tracker.active_tracks
        assert deployment_id in tracker.evidence_items
        
        flight_path = tracker.active_tracks[drone_id]
        assert len(flight_path.waypoints) == 1
        assert flight_path.waypoints[0] == initial_location
    
    @pytest.mark.asyncio
    async def test_update_position(self, tracker, initial_location):
        """Test position updates and distance calculation"""
        drone_id = "drone-001"
        deployment_id = "deployment-123"
        
        await tracker.start_tracking(drone_id, deployment_id, initial_location)
        
        # Update position
        new_location = GPSCoordinate(
            latitude=28.6140,
            longitude=77.2091,
            altitude=105.0,
            timestamp=datetime.now(),
            accuracy=3.0
        )
        
        await tracker.update_position(drone_id, new_location)
        
        flight_path = tracker.active_tracks[drone_id]
        assert len(flight_path.waypoints) == 2
        assert flight_path.total_distance > 0
        assert flight_path.max_altitude == 105.0
    
    @pytest.mark.asyncio
    async def test_stop_tracking(self, tracker, initial_location):
        """Test stopping drone tracking"""
        drone_id = "drone-001"
        deployment_id = "deployment-123"
        
        await tracker.start_tracking(drone_id, deployment_id, initial_location)
        await tracker.stop_tracking(drone_id)
        
        assert drone_id not in tracker.active_tracks

class TestEvidenceCollector:
    """Test evidence collection functionality"""
    
    @pytest.fixture
    def tracker(self):
        """Create drone tracker"""
        return DroneTracker()
    
    @pytest.fixture
    def evidence_collector(self, tracker):
        """Create evidence collector"""
        return EvidenceCollector(tracker)
    
    @pytest.fixture
    def initial_location(self):
        """Create initial GPS location"""
        return GPSCoordinate(
            latitude=28.6139,
            longitude=77.2090,
            altitude=100.0,
            timestamp=datetime.now(),
            accuracy=3.0
        )
    
    @pytest.mark.asyncio
    async def test_collect_video_evidence(self, evidence_collector, tracker, initial_location):
        """Test video evidence collection"""
        drone_id = "drone-001"
        deployment_id = "deployment-123"
        
        # Start tracking
        await tracker.start_tracking(drone_id, deployment_id, initial_location)
        
        # Start evidence processor
        await evidence_collector.start_collection_processor()
        
        try:
            # Collect evidence
            evidence_id = await evidence_collector.collect_evidence(
                drone_id, deployment_id, EvidenceType.VIDEO
            )
            
            assert evidence_id is not None
            evidence_items = tracker.get_evidence_items(deployment_id)
            assert len(evidence_items) == 1
            assert evidence_items[0].type == EvidenceType.VIDEO
            
            # Wait for processing
            await asyncio.sleep(1)
            
        finally:
            await evidence_collector.stop_collection_processor()
    
    @pytest.mark.asyncio
    async def test_collect_multiple_evidence_types(self, evidence_collector, tracker, initial_location):
        """Test collecting multiple types of evidence"""
        drone_id = "drone-001"
        deployment_id = "deployment-123"
        
        await tracker.start_tracking(drone_id, deployment_id, initial_location)
        await evidence_collector.start_collection_processor()
        
        try:
            # Collect different types of evidence
            evidence_types = [EvidenceType.VIDEO, EvidenceType.PHOTO, EvidenceType.THERMAL_IMAGE]
            evidence_ids = []
            
            for evidence_type in evidence_types:
                evidence_id = await evidence_collector.collect_evidence(
                    drone_id, deployment_id, evidence_type
                )
                evidence_ids.append(evidence_id)
            
            evidence_items = tracker.get_evidence_items(deployment_id)
            assert len(evidence_items) == 3
            
            # Check all types are present
            collected_types = {item.type for item in evidence_items}
            assert collected_types == set(evidence_types)
            
            # Wait for processing
            await asyncio.sleep(1)
            
        finally:
            await evidence_collector.stop_collection_processor()
    
    @pytest.mark.asyncio
    async def test_evidence_summary(self, evidence_collector, tracker, initial_location):
        """Test evidence collection summary"""
        drone_id = "drone-001"
        deployment_id = "deployment-123"
        
        await tracker.start_tracking(drone_id, deployment_id, initial_location)
        await evidence_collector.start_collection_processor()
        
        try:
            # Collect evidence
            await evidence_collector.collect_evidence(
                drone_id, deployment_id, EvidenceType.VIDEO
            )
            await evidence_collector.collect_evidence(
                drone_id, deployment_id, EvidenceType.PHOTO
            )
            
            # Get summary
            summary = await evidence_collector.get_evidence_summary(deployment_id)
            
            assert summary["total_items"] == 2
            assert summary["by_type"]["video"] == 1
            assert summary["by_type"]["photo"] == 1
            assert summary["total_size_estimate"] > 0
            
        finally:
            await evidence_collector.stop_collection_processor()

class TestHandoffProtocols:
    """Test drone handoff protocols"""
    
    @pytest.fixture
    def handoff_protocol(self):
        """Create handoff protocol instance"""
        return DroneHandoffProtocol()
    
    @pytest.mark.asyncio
    async def test_initiate_operator_handoff(self, handoff_protocol):
        """Test initiating operator-to-operator handoff"""
        handoff_id = await handoff_protocol.initiate_handoff(
            drone_id="drone-001",
            deployment_id="deployment-123",
            from_operator_id="op-001",
            handoff_type=HandoffType.OPERATOR_TO_OPERATOR,
            reason="Shift change",
            urgency="medium"
        )
        
        assert handoff_id is not None
        assert handoff_id in handoff_protocol.active_handoffs
        
        handoff = handoff_protocol.active_handoffs[handoff_id]
        assert handoff.drone_id == "drone-001"
        assert handoff.from_operator_id == "op-001"
        assert handoff.handoff_type == HandoffType.OPERATOR_TO_OPERATOR
    
    @pytest.mark.asyncio
    async def test_respond_to_handoff_accepted(self, handoff_protocol):
        """Test accepting handoff request"""
        handoff_id = await handoff_protocol.initiate_handoff(
            drone_id="drone-001",
            deployment_id="deployment-123",
            from_operator_id="op-001",
            handoff_type=HandoffType.OPERATOR_TO_OPERATOR,
            reason="Shift change",
            to_operator_id="op-002"
        )
        
        result = await handoff_protocol.respond_to_handoff(
            handoff_id=handoff_id,
            operator_id="op-002",
            accepted=True,
            response_reason="Ready to take over"
        )
        
        assert result is True
        assert handoff_id not in handoff_protocol.active_handoffs  # Moved to history
        
        # Check in history
        history = handoff_protocol.get_handoff_history()
        completed_handoff = next((h for h in history if h.id == handoff_id), None)
        assert completed_handoff is not None
        assert completed_handoff.status.value == "completed"
    
    @pytest.mark.asyncio
    async def test_respond_to_handoff_rejected(self, handoff_protocol):
        """Test rejecting handoff request"""
        handoff_id = await handoff_protocol.initiate_handoff(
            drone_id="drone-001",
            deployment_id="deployment-123",
            from_operator_id="op-001",
            handoff_type=HandoffType.OPERATOR_TO_OPERATOR,
            reason="Shift change",
            to_operator_id="op-002"
        )
        
        result = await handoff_protocol.respond_to_handoff(
            handoff_id=handoff_id,
            operator_id="op-002",
            accepted=False,
            response_reason="Currently handling another incident"
        )
        
        assert result is False
        handoff = handoff_protocol.active_handoffs[handoff_id]
        assert handoff.status.value == "rejected"
    
    @pytest.mark.asyncio
    async def test_emergency_override(self, handoff_protocol):
        """Test emergency override functionality"""
        handoff_id = await handoff_protocol.emergency_override(
            drone_id="drone-001",
            deployment_id="deployment-123",
            commander_id="sup-001",  # Supervisor has commander-level access in mock data
            override_reason="Critical security threat"
        )
        
        assert handoff_id is not None
        
        # Check in history (should be auto-completed)
        history = handoff_protocol.get_handoff_history()
        override_handoff = next((h for h in history if h.id == handoff_id), None)
        assert override_handoff is not None
        assert override_handoff.handoff_type == HandoffType.EMERGENCY_OVERRIDE
        assert override_handoff.status.value == "completed"
    
    @pytest.mark.asyncio
    async def test_cancel_handoff(self, handoff_protocol):
        """Test cancelling handoff request"""
        handoff_id = await handoff_protocol.initiate_handoff(
            drone_id="drone-001",
            deployment_id="deployment-123",
            from_operator_id="op-001",
            handoff_type=HandoffType.OPERATOR_TO_OPERATOR,
            reason="Shift change"
        )
        
        result = await handoff_protocol.cancel_handoff(
            handoff_id=handoff_id,
            operator_id="op-001",
            reason="Incident resolved"
        )
        
        assert result is True
        assert handoff_id not in handoff_protocol.active_handoffs
        
        # Check in history
        history = handoff_protocol.get_handoff_history()
        cancelled_handoff = next((h for h in history if h.id == handoff_id), None)
        assert cancelled_handoff is not None
        assert cancelled_handoff.status.value == "cancelled"
    
    def test_find_suitable_operators(self, handoff_protocol):
        """Test finding suitable operators for handoff"""
        # This tests the internal logic for operator matching
        # In a real implementation, this would test against actual operator data
        
        # Mock handoff request
        from handoff_protocols import HandoffRequest
        handoff_request = HandoffRequest(
            id="test-handoff",
            handoff_type=HandoffType.OPERATOR_TO_OPERATOR,
            drone_id="drone-001",
            deployment_id="deployment-123",
            from_operator_id="op-001",
            to_operator_id=None,
            authorization_level=AuthorizationLevel.OPERATOR,
            reason="Test",
            urgency="medium",
            requested_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=15),
            metadata={}
        )
        
        # Test that the method exists and returns a list
        suitable_operators = asyncio.run(
            handoff_protocol._find_suitable_operators(handoff_request)
        )
        assert isinstance(suitable_operators, list)

class TestIntegrationWorkflows:
    """Test end-to-end integration workflows"""
    
    @pytest.fixture
    def drone_service(self):
        """Create drone service for integration tests"""
        return DroneIntegrationService()
    
    @pytest.mark.asyncio
    async def test_complete_deployment_workflow(self, drone_service):
        """Test complete deployment workflow from request to completion"""
        # 1. Request deployment
        deployment_request = DeploymentRequest(
            incident_id="incident-123",
            target_location={"lat": 28.6139, "lon": 77.2090},
            mission_type="surveillance",
            priority="high",
            estimated_duration=30,
            required_capabilities=["surveillance"],
            operator_notes="Border crossing detected"
        )
        
        deployment_id = await drone_service.request_drone_deployment(
            deployment_request, "operator-123"
        )
        
        # 2. Authorize deployment
        auth_request = AuthorizationRequest(
            deployment_id=deployment_id,
            operator_id="supervisor-001",
            authorization_reason="Valid threat",
            approved=True
        )
        
        await drone_service.authorize_deployment(auth_request)
        
        # 3. Collect evidence
        evidence_id = await drone_service.collect_evidence(deployment_id, "video")
        
        # 4. Recall drone
        await drone_service.recall_drone(deployment_id, "operator-123")
        
        # Verify final state
        deployment = drone_service.deployments[deployment_id]
        assert deployment.status.value == "completed"
        assert len(deployment.evidence_collected) == 1
        assert evidence_id in deployment.evidence_collected
    
    @pytest.mark.asyncio
    async def test_handoff_during_mission(self, drone_service):
        """Test handoff during active mission"""
        # Start deployment
        deployment_request = DeploymentRequest(
            incident_id="incident-123",
            target_location={"lat": 28.6139, "lon": 77.2090},
            mission_type="surveillance",
            priority="high",
            estimated_duration=60,
            required_capabilities=["surveillance"]
        )
        
        deployment_id = await drone_service.request_drone_deployment(
            deployment_request, "op-001"
        )
        
        auth_request = AuthorizationRequest(
            deployment_id=deployment_id,
            operator_id="supervisor-001",
            authorization_reason="Approved",
            approved=True
        )
        
        await drone_service.authorize_deployment(auth_request)
        
        # Initiate handoff
        handoff_id = await drone_service.handoff_protocol.initiate_handoff(
            drone_id=drone_service.deployments[deployment_id].drone_id,
            deployment_id=deployment_id,
            from_operator_id="op-001",
            handoff_type=HandoffType.OPERATOR_TO_OPERATOR,
            reason="Shift change",
            to_operator_id="sup-001"
        )
        
        # Accept handoff
        await drone_service.handoff_protocol.respond_to_handoff(
            handoff_id=handoff_id,
            operator_id="sup-001",
            accepted=True,
            response_reason="Taking over"
        )
        
        # Verify handoff completed
        history = drone_service.handoff_protocol.get_handoff_history()
        completed_handoff = next((h for h in history if h.id == handoff_id), None)
        assert completed_handoff is not None
        assert completed_handoff.status.value == "completed"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])