"""
Alert Service - Main entry point for Project Argus Alert Management
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from alert_engine import AlertEngine
from incident_manager import IncidentManager
from notification_service import NotificationService, EscalationService
from shared.models import Alert, CrossingEvent, Detection, Incident
from shared.models.alerts import AlertType, Severity
from shared.models.incidents import IncidentStatus, Resolution


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
alert_engine: AlertEngine = None
incident_manager: IncidentManager = None
notification_service: NotificationService = None
escalation_service: EscalationService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global alert_engine, incident_manager, notification_service, escalation_service
    
    logger.info("Starting Alert Service...")
    
    # Initialize services
    alert_engine = AlertEngine()
    incident_manager = IncidentManager()
    notification_service = NotificationService()
    escalation_service = EscalationService(notification_service)
    
    # Setup alert routing to create incidents for high-severity alerts
    def handle_high_severity_alert(alert: Alert):
        if alert.severity in [Severity.HIGH, Severity.CRITICAL]:
            incident = incident_manager.create_incident(
                alert, 
                "system",
                f"Auto-generated incident for {alert.type.value}"
            )
            logger.info(f"Created incident {incident.id} for alert {alert.id}")
            
            # Process escalation
            escalation_service.process_escalation(alert)
    
    alert_engine.register_alert_handler(handle_high_severity_alert)
    
    logger.info("Alert Service initialized successfully")
    yield
    
    logger.info("Shutting down Alert Service...")


# Create FastAPI application
app = FastAPI(
    title="Project Argus Alert Service",
    description="Alert generation and management service for border detection system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get services
def get_alert_engine() -> AlertEngine:
    return alert_engine

def get_incident_manager() -> IncidentManager:
    return incident_manager

def get_notification_service() -> NotificationService:
    return notification_service

def get_escalation_service() -> EscalationService:
    return escalation_service


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "alert-service"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Project Argus Alert Service", "version": "1.0.0"}


# Alert endpoints
@app.post("/alerts/generate")
async def generate_alert(
    crossing_event: dict,
    detection: dict,
    engine: AlertEngine = Depends(get_alert_engine)
):
    """Generate an alert from crossing event and detection data."""
    try:
        # Convert dict to objects (simplified for demo)
        crossing_obj = CrossingEvent(**crossing_event)
        detection_obj = Detection(**detection)
        
        alert = engine.generate_alert(crossing_obj, detection_obj)
        engine.route_alert(alert)
        
        return {"alert_id": alert.id, "status": "generated", "alert": alert.to_dict()}
    except Exception as e:
        logger.error(f"Error generating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts/active")
async def get_active_alerts(
    camera_id: str = None,
    engine: AlertEngine = Depends(get_alert_engine)
):
    """Get active alerts, optionally filtered by camera."""
    try:
        alerts = engine.get_active_alerts(camera_id)
        return {
            "alerts": [alert.to_dict() for alert in alerts],
            "count": len(alerts)
        }
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    operator_id: str,
    engine: AlertEngine = Depends(get_alert_engine)
):
    """Acknowledge an alert."""
    try:
        success = engine.acknowledge_alert(alert_id, operator_id)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"status": "acknowledged", "alert_id": alert_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alerts/{alert_id}/escalate")
async def escalate_alert(
    alert_id: str,
    operator_id: str,
    reason: str,
    engine: AlertEngine = Depends(get_alert_engine)
):
    """Escalate an alert."""
    try:
        success = engine.escalate_alert(alert_id, operator_id, reason)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"status": "escalated", "alert_id": alert_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error escalating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts/statistics")
async def get_alert_statistics(engine: AlertEngine = Depends(get_alert_engine)):
    """Get alert engine statistics."""
    try:
        stats = engine.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting alert statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Incident endpoints
@app.post("/incidents/create")
async def create_incident(
    alert_id: str,
    operator_id: str,
    title: str = None,
    description: str = None,
    manager: IncidentManager = Depends(get_incident_manager),
    engine: AlertEngine = Depends(get_alert_engine)
):
    """Create an incident from an alert."""
    try:
        alert = engine.active_alerts.get(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        incident = manager.create_incident(alert, operator_id, title, description)
        return {"incident_id": incident.id, "status": "created", "incident": incident.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/incidents/{incident_id}")
async def get_incident(
    incident_id: str,
    manager: IncidentManager = Depends(get_incident_manager)
):
    """Get incident by ID."""
    try:
        incident = manager.get_incident(incident_id)
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        return incident.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/incidents")
async def get_incidents(
    status: str = None,
    operator_id: str = None,
    manager: IncidentManager = Depends(get_incident_manager)
):
    """Get incidents by status or operator."""
    try:
        if status:
            status_enum = IncidentStatus(status)
            incidents = manager.get_incidents_by_status(status_enum)
        elif operator_id:
            incidents = manager.get_incidents_by_operator(operator_id)
        else:
            incidents = list(manager.incidents.values())
        
        return {
            "incidents": [incident.to_dict() for incident in incidents],
            "count": len(incidents)
        }
    except Exception as e:
        logger.error(f"Error getting incidents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/incidents/{incident_id}/assign")
async def assign_incident(
    incident_id: str,
    operator_id: str,
    manager: IncidentManager = Depends(get_incident_manager)
):
    """Assign incident to operator."""
    try:
        success = manager.assign_incident(incident_id, operator_id)
        if not success:
            raise HTTPException(status_code=404, detail="Incident not found")
        return {"status": "assigned", "incident_id": incident_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/incidents/{incident_id}/resolve")
async def resolve_incident(
    incident_id: str,
    resolution: dict,
    manager: IncidentManager = Depends(get_incident_manager)
):
    """Resolve an incident."""
    try:
        resolution_obj = Resolution(**resolution)
        success = manager.resolve_incident(incident_id, resolution_obj)
        if not success:
            raise HTTPException(status_code=404, detail="Incident not found")
        return {"status": "resolved", "incident_id": incident_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/incidents/statistics")
async def get_incident_statistics(
    time_range_hours: int = 24,
    manager: IncidentManager = Depends(get_incident_manager)
):
    """Get incident statistics."""
    try:
        stats = manager.get_incident_statistics(time_range_hours)
        return stats
    except Exception as e:
        logger.error(f"Error getting incident statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Notification endpoints
@app.post("/notifications/send")
async def send_notification(
    alert_id: str,
    recipients: List[str],
    channels: List[str],
    service: NotificationService = Depends(get_notification_service),
    engine: AlertEngine = Depends(get_alert_engine)
):
    """Send notification for an alert."""
    try:
        alert = engine.active_alerts.get(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        results = service.notify_alert(alert, recipients, channels)
        return {"status": "sent", "results": results}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )