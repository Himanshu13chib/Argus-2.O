"""
Drone Handoff Protocols
Manages drone handoff between operators and automated systems with human authorization
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import uuid

logger = logging.getLogger(__name__)

class HandoffType(str, Enum):
    OPERATOR_TO_OPERATOR = "operator_to_operator"
    SYSTEM_TO_OPERATOR = "system_to_operator"
    OPERATOR_TO_SYSTEM = "operator_to_system"
    EMERGENCY_OVERRIDE = "emergency_override"

class HandoffStatus(str, Enum):
    INITIATED = "initiated"
    PENDING_ACCEPTANCE = "pending_acceptance"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class AuthorizationLevel(str, Enum):
    OPERATOR = "operator"
    SUPERVISOR = "supervisor"
    COMMANDER = "commander"
    EMERGENCY = "emergency"

@dataclass
class HandoffRequest:
    id: str
    handoff_type: HandoffType
    drone_id: str
    deployment_id: str
    from_operator_id: str
    to_operator_id: Optional[str]
    authorization_level: AuthorizationLevel
    reason: str
    urgency: str  # low, medium, high, critical
    requested_at: datetime
    expires_at: datetime
    metadata: Dict
    status: HandoffStatus = HandoffStatus.INITIATED

@dataclass
class HandoffResponse:
    handoff_id: str
    operator_id: str
    accepted: bool
    response_reason: str
    responded_at: datetime
    conditions: Optional[List[str]] = None

@dataclass
class OperatorCapability:
    operator_id: str
    drone_types: List[str]
    mission_types: List[str]
    authorization_level: AuthorizationLevel
    current_workload: int
    max_concurrent_drones: int
    availability_status: str
    last_active: datetime

class DroneHandoffProtocol:
    """Manages drone handoff protocols with human authorization"""
    
    def __init__(self):
        self.active_handoffs: Dict[str, HandoffRequest] = {}
        self.operator_capabilities: Dict[str, OperatorCapability] = {}
        self.handoff_history: List[HandoffRequest] = []
        self.notification_callbacks: List[Callable] = []
        self._monitoring_task = None
        self._initialize_mock_operators()
    
    async def start_monitoring(self):
        """Start background monitoring tasks"""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_handoff_expiry())
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
    
    def _initialize_mock_operators(self):
        """Initialize mock operator capabilities"""
        mock_operators = [
            {
                "operator_id": "op-001",
                "drone_types": ["surveillance", "tracking"],
                "mission_types": ["border_patrol", "incident_response"],
                "authorization_level": AuthorizationLevel.OPERATOR,
                "current_workload": 1,
                "max_concurrent_drones": 3,
                "availability_status": "available",
                "last_active": datetime.now()
            },
            {
                "operator_id": "sup-001",
                "drone_types": ["surveillance", "tracking", "tactical"],
                "mission_types": ["border_patrol", "incident_response", "emergency"],
                "authorization_level": AuthorizationLevel.SUPERVISOR,
                "current_workload": 0,
                "max_concurrent_drones": 5,
                "availability_status": "available",
                "last_active": datetime.now()
            }
        ]
        
        for op_data in mock_operators:
            capability = OperatorCapability(**op_data)
            self.operator_capabilities[capability.operator_id] = capability
    
    async def initiate_handoff(
        self,
        drone_id: str,
        deployment_id: str,
        from_operator_id: str,
        handoff_type: HandoffType,
        reason: str,
        urgency: str = "medium",
        to_operator_id: Optional[str] = None,
        authorization_level: AuthorizationLevel = AuthorizationLevel.OPERATOR
    ) -> str:
        """Initiate a drone handoff request"""
        
        handoff_id = str(uuid.uuid4())
        
        # Calculate expiry based on urgency
        expiry_minutes = {
            "low": 30,
            "medium": 15,
            "high": 5,
            "critical": 2
        }
        
        expires_at = datetime.now() + timedelta(minutes=expiry_minutes.get(urgency, 15))
        
        handoff_request = HandoffRequest(
            id=handoff_id,
            handoff_type=handoff_type,
            drone_id=drone_id,
            deployment_id=deployment_id,
            from_operator_id=from_operator_id,
            to_operator_id=to_operator_id,
            authorization_level=authorization_level,
            reason=reason,
            urgency=urgency,
            requested_at=datetime.now(),
            expires_at=expires_at,
            metadata={},
            status=HandoffStatus.INITIATED
        )
        
        self.active_handoffs[handoff_id] = handoff_request
        
        logger.info(f"Handoff initiated: {handoff_id} for drone {drone_id}")
        
        # Find suitable operators if not specified
        if not to_operator_id and handoff_type != HandoffType.OPERATOR_TO_SYSTEM:
            suitable_operators = await self._find_suitable_operators(handoff_request)
            if suitable_operators:
                handoff_request.status = HandoffStatus.PENDING_ACCEPTANCE
                await self._notify_operators(handoff_request, suitable_operators)
            else:
                logger.warning(f"No suitable operators found for handoff {handoff_id}")
        elif to_operator_id:
            handoff_request.to_operator_id = to_operator_id
            handoff_request.status = HandoffStatus.PENDING_ACCEPTANCE
            await self._notify_specific_operator(handoff_request, to_operator_id)
        else:
            # System handoff - auto-accept if authorized
            if await self._validate_system_handoff(handoff_request):
                handoff_request.status = HandoffStatus.ACCEPTED
                await self._execute_handoff(handoff_request)
        
        return handoff_id
    
    async def respond_to_handoff(
        self,
        handoff_id: str,
        operator_id: str,
        accepted: bool,
        response_reason: str,
        conditions: Optional[List[str]] = None
    ) -> bool:
        """Respond to a handoff request"""
        
        handoff_request = self.active_handoffs.get(handoff_id)
        if not handoff_request:
            raise ValueError(f"Handoff request {handoff_id} not found")
        
        if handoff_request.status != HandoffStatus.PENDING_ACCEPTANCE:
            raise ValueError(f"Handoff {handoff_id} is not pending acceptance")
        
        if datetime.now() > handoff_request.expires_at:
            handoff_request.status = HandoffStatus.EXPIRED
            raise ValueError(f"Handoff {handoff_id} has expired")
        
        response = HandoffResponse(
            handoff_id=handoff_id,
            operator_id=operator_id,
            accepted=accepted,
            response_reason=response_reason,
            responded_at=datetime.now(),
            conditions=conditions
        )
        
        if accepted:
            handoff_request.status = HandoffStatus.ACCEPTED
            handoff_request.to_operator_id = operator_id
            
            logger.info(f"Handoff {handoff_id} accepted by operator {operator_id}")
            
            # Execute the handoff
            await self._execute_handoff(handoff_request)
            
        else:
            handoff_request.status = HandoffStatus.REJECTED
            logger.info(f"Handoff {handoff_id} rejected by operator {operator_id}: {response_reason}")
            
            # Try to find alternative operators
            await self._handle_handoff_rejection(handoff_request)
        
        return accepted
    
    async def cancel_handoff(self, handoff_id: str, operator_id: str, reason: str) -> bool:
        """Cancel an active handoff request"""
        
        handoff_request = self.active_handoffs.get(handoff_id)
        if not handoff_request:
            return False
        
        if handoff_request.from_operator_id != operator_id:
            raise ValueError("Only the requesting operator can cancel the handoff")
        
        handoff_request.status = HandoffStatus.CANCELLED
        handoff_request.metadata["cancellation_reason"] = reason
        handoff_request.metadata["cancelled_at"] = datetime.now()
        
        logger.info(f"Handoff {handoff_id} cancelled by operator {operator_id}: {reason}")
        
        # Move to history
        self.handoff_history.append(handoff_request)
        del self.active_handoffs[handoff_id]
        
        return True
    
    async def emergency_override(
        self,
        drone_id: str,
        deployment_id: str,
        commander_id: str,
        override_reason: str
    ) -> str:
        """Execute emergency override handoff"""
        
        # Verify commander authorization
        commander_capability = self.operator_capabilities.get(commander_id)
        if not commander_capability or commander_capability.authorization_level != AuthorizationLevel.COMMANDER:
            raise ValueError("Insufficient authorization for emergency override")
        
        handoff_id = await self.initiate_handoff(
            drone_id=drone_id,
            deployment_id=deployment_id,
            from_operator_id="system",
            handoff_type=HandoffType.EMERGENCY_OVERRIDE,
            reason=override_reason,
            urgency="critical",
            to_operator_id=commander_id,
            authorization_level=AuthorizationLevel.EMERGENCY
        )
        
        # Auto-accept emergency override
        handoff_request = self.active_handoffs[handoff_id]
        handoff_request.status = HandoffStatus.ACCEPTED
        await self._execute_handoff(handoff_request)
        
        logger.warning(f"Emergency override executed: {handoff_id} by commander {commander_id}")
        
        return handoff_id
    
    async def _find_suitable_operators(self, handoff_request: HandoffRequest) -> List[str]:
        """Find operators suitable for the handoff"""
        suitable_operators = []
        
        for operator_id, capability in self.operator_capabilities.items():
            # Skip the requesting operator
            if operator_id == handoff_request.from_operator_id:
                continue
            
            # Check availability
            if capability.availability_status != "available":
                continue
            
            # Check workload
            if capability.current_workload >= capability.max_concurrent_drones:
                continue
            
            # Check authorization level
            if capability.authorization_level.value < handoff_request.authorization_level.value:
                continue
            
            # Check if operator was active recently
            if (datetime.now() - capability.last_active).total_seconds() > 3600:  # 1 hour
                continue
            
            suitable_operators.append(operator_id)
        
        # Sort by workload (prefer less busy operators)
        suitable_operators.sort(
            key=lambda op_id: self.operator_capabilities[op_id].current_workload
        )
        
        return suitable_operators
    
    async def _validate_system_handoff(self, handoff_request: HandoffRequest) -> bool:
        """Validate system handoff authorization"""
        # Check if system handoff is authorized
        if handoff_request.handoff_type == HandoffType.OPERATOR_TO_SYSTEM:
            # Require supervisor approval for system handoffs
            return handoff_request.authorization_level in [
                AuthorizationLevel.SUPERVISOR,
                AuthorizationLevel.COMMANDER
            ]
        
        return True
    
    async def _execute_handoff(self, handoff_request: HandoffRequest):
        """Execute the actual handoff"""
        try:
            # Update operator workloads
            if handoff_request.from_operator_id in self.operator_capabilities:
                self.operator_capabilities[handoff_request.from_operator_id].current_workload -= 1
            
            if handoff_request.to_operator_id and handoff_request.to_operator_id in self.operator_capabilities:
                self.operator_capabilities[handoff_request.to_operator_id].current_workload += 1
            
            # Mark as completed
            handoff_request.status = HandoffStatus.COMPLETED
            handoff_request.metadata["completed_at"] = datetime.now()
            
            # Notify relevant systems
            await self._notify_handoff_completion(handoff_request)
            
            # Move to history
            self.handoff_history.append(handoff_request)
            del self.active_handoffs[handoff_request.id]
            
            logger.info(f"Handoff {handoff_request.id} completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to execute handoff {handoff_request.id}: {e}")
            handoff_request.status = HandoffStatus.REJECTED
            handoff_request.metadata["error"] = str(e)
    
    async def _handle_handoff_rejection(self, handoff_request: HandoffRequest):
        """Handle handoff rejection and find alternatives"""
        # Try to find alternative operators
        suitable_operators = await self._find_suitable_operators(handoff_request)
        
        if suitable_operators:
            handoff_request.status = HandoffStatus.PENDING_ACCEPTANCE
            await self._notify_operators(handoff_request, suitable_operators)
        else:
            # Escalate to supervisor
            await self._escalate_handoff(handoff_request)
    
    async def _escalate_handoff(self, handoff_request: HandoffRequest):
        """Escalate handoff to higher authority"""
        supervisors = [
            op_id for op_id, capability in self.operator_capabilities.items()
            if capability.authorization_level in [AuthorizationLevel.SUPERVISOR, AuthorizationLevel.COMMANDER]
        ]
        
        if supervisors:
            handoff_request.authorization_level = AuthorizationLevel.SUPERVISOR
            handoff_request.urgency = "high"
            handoff_request.expires_at = datetime.now() + timedelta(minutes=5)
            handoff_request.status = HandoffStatus.PENDING_ACCEPTANCE
            
            await self._notify_operators(handoff_request, supervisors)
            logger.warning(f"Handoff {handoff_request.id} escalated to supervisors")
    
    async def _notify_operators(self, handoff_request: HandoffRequest, operator_ids: List[str]):
        """Notify operators about handoff request"""
        for callback in self.notification_callbacks:
            try:
                await callback("handoff_request", {
                    "handoff_id": handoff_request.id,
                    "drone_id": handoff_request.drone_id,
                    "operator_ids": operator_ids,
                    "urgency": handoff_request.urgency,
                    "reason": handoff_request.reason
                })
            except Exception as e:
                logger.error(f"Failed to notify operators: {e}")
    
    async def _notify_specific_operator(self, handoff_request: HandoffRequest, operator_id: str):
        """Notify specific operator about handoff request"""
        await self._notify_operators(handoff_request, [operator_id])
    
    async def _notify_handoff_completion(self, handoff_request: HandoffRequest):
        """Notify systems about handoff completion"""
        for callback in self.notification_callbacks:
            try:
                await callback("handoff_completed", {
                    "handoff_id": handoff_request.id,
                    "drone_id": handoff_request.drone_id,
                    "from_operator": handoff_request.from_operator_id,
                    "to_operator": handoff_request.to_operator_id,
                    "handoff_type": handoff_request.handoff_type
                })
            except Exception as e:
                logger.error(f"Failed to notify handoff completion: {e}")
    
    async def _monitor_handoff_expiry(self):
        """Monitor and handle expired handoff requests"""
        while True:
            try:
                current_time = datetime.now()
                expired_handoffs = []
                
                for handoff_id, handoff_request in self.active_handoffs.items():
                    if (handoff_request.status == HandoffStatus.PENDING_ACCEPTANCE and 
                        current_time > handoff_request.expires_at):
                        expired_handoffs.append(handoff_id)
                
                for handoff_id in expired_handoffs:
                    handoff_request = self.active_handoffs[handoff_id]
                    handoff_request.status = HandoffStatus.EXPIRED
                    
                    logger.warning(f"Handoff {handoff_id} expired")
                    
                    # Try to escalate or find alternatives
                    await self._escalate_handoff(handoff_request)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring handoff expiry: {e}")
                await asyncio.sleep(30)
    
    def add_notification_callback(self, callback: Callable):
        """Add notification callback for handoff events"""
        self.notification_callbacks.append(callback)
    
    def get_active_handoffs(self) -> List[HandoffRequest]:
        """Get all active handoff requests"""
        return list(self.active_handoffs.values())
    
    def get_handoff_history(self, limit: int = 100) -> List[HandoffRequest]:
        """Get handoff history"""
        return self.handoff_history[-limit:]
    
    def get_operator_workload(self, operator_id: str) -> Optional[OperatorCapability]:
        """Get operator workload and capability info"""
        return self.operator_capabilities.get(operator_id)
    
    def update_operator_availability(self, operator_id: str, status: str):
        """Update operator availability status"""
        if operator_id in self.operator_capabilities:
            self.operator_capabilities[operator_id].availability_status = status
            self.operator_capabilities[operator_id].last_active = datetime.now()