#!/usr/bin/env python3
"""
Simplified API Gateway for Project Argus - Development Mode
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Project Argus API Gateway",
    description="Central API Gateway for Project Argus Border Detection System",
    version="1.0.0"
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data for demonstration
mock_cameras = [
    {
        "id": "camera_001",
        "name": "Border Sector Alpha - Main Gate",
        "location": {"lat": 28.6139, "lng": 77.2090},
        "status": "active",
        "type": "visible_thermal",
        "resolution": "1920x1080",
        "fps": 30
    },
    {
        "id": "camera_002", 
        "name": "Border Sector Alpha - Perimeter East",
        "location": {"lat": 28.6140, "lng": 77.2095},
        "status": "active",
        "type": "thermal_night_vision",
        "resolution": "1280x720",
        "fps": 25
    },
    {
        "id": "camera_003",
        "name": "Border Sector Beta - Patrol Route",
        "location": {"lat": 28.6135, "lng": 77.2085},
        "status": "active",
        "type": "visible_ptz",
        "resolution": "1920x1080",
        "fps": 30
    },
    {
        "id": "camera_004",
        "name": "Border Sector Beta - Remote Outpost",
        "location": {"lat": 28.6130, "lng": 77.2080},
        "status": "maintenance",
        "type": "thermal_radar",
        "resolution": "640x480",
        "fps": 15
    }
]

mock_alerts = [
    {
        "id": "alert_001",
        "type": "virtual_line_crossing",
        "severity": "high",
        "camera_id": "camera_001",
        "timestamp": datetime.now().isoformat(),
        "confidence": 0.94,
        "description": "Unauthorized crossing detected at main gate",
        "acknowledged": False
    },
    {
        "id": "alert_002",
        "type": "motion_detection",
        "severity": "medium",
        "camera_id": "camera_002",
        "timestamp": datetime.now().isoformat(),
        "confidence": 0.78,
        "description": "Motion detected in perimeter zone",
        "acknowledged": True
    }
]

mock_incidents = [
    {
        "id": "incident_001",
        "alert_id": "alert_001",
        "status": "open",
        "priority": "high",
        "created_at": datetime.now().isoformat(),
        "description": "Border crossing investigation ongoing",
        "assigned_to": "operator_001"
    }
]

mock_detections = [
    {
        "id": "detection_001",
        "camera_id": "camera_001",
        "timestamp": datetime.now().isoformat(),
        "confidence": 0.94,
        "bbox": {"x": 100, "y": 100, "width": 50, "height": 100},
        "detection_class": "person"
    }
]


@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "message": "Project Argus API Gateway", 
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "cameras": f"/api/cameras ({len(mock_cameras)} active)",
            "alerts": f"/api/alerts ({len([a for a in mock_alerts if not a['acknowledged']])} unacknowledged)",
            "incidents": f"/api/incidents ({len([i for i in mock_incidents if i['status'] == 'open'])} open)",
            "detections": f"/api/detections"
        },
        "endpoints": {
            "health": "/health",
            "dashboard": "/dashboard",
            "api_docs": "/docs",
            "cameras": "/api/cameras",
            "alerts": "/api/alerts",
            "incidents": "/api/incidents",
            "detections": "/api/detections"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "api-gateway",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "uptime": "running",
        "dependencies": {
            "database": "connected",
            "cache": "connected",
            "storage": "connected"
        }
    }


@app.get("/api/cameras")
async def get_cameras():
    """Get all cameras."""
    return {
        "cameras": mock_cameras,
        "total": len(mock_cameras),
        "active": len([c for c in mock_cameras if c["status"] == "active"]),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/cameras/{camera_id}")
async def get_camera(camera_id: str):
    """Get specific camera."""
    camera = next((c for c in mock_cameras if c["id"] == camera_id), None)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@app.get("/api/alerts")
async def get_alerts():
    """Get all alerts."""
    return {
        "alerts": mock_alerts,
        "total": len(mock_alerts),
        "unacknowledged": len([a for a in mock_alerts if not a["acknowledged"]]),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    alert = next((a for a in mock_alerts if a["id"] == alert_id), None)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert["acknowledged"] = True
    alert["acknowledged_at"] = datetime.now().isoformat()
    
    return {"message": "Alert acknowledged", "alert": alert}


@app.get("/api/incidents")
async def get_incidents():
    """Get all incidents."""
    return {
        "incidents": mock_incidents,
        "total": len(mock_incidents),
        "open": len([i for i in mock_incidents if i["status"] == "open"]),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/detections")
async def get_detections():
    """Get recent detections."""
    return {
        "detections": mock_detections,
        "total": len(mock_detections),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/system/status")
async def get_system_status():
    """Get system status."""
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "detection_pipeline": "active",
            "alert_system": "active", 
            "evidence_store": "active",
            "tracking_service": "active"
        },
        "metrics": {
            "cameras_active": len([c for c in mock_cameras if c["status"] == "active"]),
            "alerts_today": len(mock_alerts),
            "incidents_open": len([i for i in mock_incidents if i["status"] == "open"]),
            "detections_today": 47,
            "system_uptime": "99.9%",
            "average_latency": "127ms"
        }
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve dashboard HTML."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Argus - Command Center</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        .header {
            background: rgba(0,0,0,0.3);
            padding: 1rem;
            text-align: center;
            border-bottom: 2px solid #4CAF50;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-top: 2rem;
        }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .status { 
            padding: 0.25rem 0.75rem; 
            border-radius: 20px; 
            font-size: 0.8rem; 
            font-weight: bold; 
        }
        .status.active { background: #4CAF50; }
        .status.alert { background: #FF5722; }
        .status.warning { background: #FF9800; }
        .btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 5px;
            cursor: pointer;
            margin: 0.5rem;
        }
        .metric {
            text-align: center;
            margin: 1rem 0;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #4CAF50;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Project Argus - Command Center</h1>
        <p>Real-time Border Security Monitoring</p>
    </div>
    
    <div class="container">
        <div class="grid">
            <div class="card">
                <h3>📊 System Metrics</h3>
                <div class="metric">
                    <div class="metric-value" id="cameras">4</div>
                    <div>Active Cameras</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="alerts">2</div>
                    <div>Active Alerts</div>
                </div>
            </div>
            
            <div class="card">
                <h3>📹 Camera Status</h3>
                <div id="camera-list">Loading...</div>
                <button class="btn" onclick="refreshCameras()">Refresh</button>
            </div>
            
            <div class="card">
                <h3>🚨 Recent Alerts</h3>
                <div id="alert-list">Loading...</div>
                <button class="btn" onclick="refreshAlerts()">Refresh</button>
            </div>
            
            <div class="card">
                <h3>🔍 System Status</h3>
                <p><span class="status active">●</span> Detection Pipeline: Active</p>
                <p><span class="status active">●</span> Alert System: Active</p>
                <p><span class="status active">●</span> Evidence Store: Active</p>
                <p><span class="status warning">●</span> Development Mode</p>
            </div>
        </div>
    </div>
    
    <script>
        async function loadData(endpoint) {
            try {
                const response = await fetch(`/api/${endpoint}`);
                return await response.json();
            } catch (error) {
                console.error(`Error loading ${endpoint}:`, error);
                return null;
            }
        }
        
        async function refreshCameras() {
            const data = await loadData('cameras');
            if (data) {
                const container = document.getElementById('camera-list');
                container.innerHTML = data.cameras.map(camera => 
                    `<div style="margin: 0.5rem 0;">
                        <strong>${camera.name}</strong><br>
                        <span class="status ${camera.status}">${camera.status}</span>
                    </div>`
                ).join('');
                document.getElementById('cameras').textContent = data.active;
            }
        }
        
        async function refreshAlerts() {
            const data = await loadData('alerts');
            if (data) {
                const container = document.getElementById('alert-list');
                container.innerHTML = data.alerts.slice(0, 3).map(alert => 
                    `<div style="margin: 0.5rem 0; padding: 0.5rem; background: rgba(255,87,34,0.2); border-radius: 5px;">
                        <strong>${alert.type.replace('_', ' ').toUpperCase()}</strong><br>
                        <small>${alert.description}</small><br>
                        <span class="status ${alert.severity}">${alert.severity}</span>
                    </div>`
                ).join('');
                document.getElementById('alerts').textContent = data.unacknowledged;
            }
        }
        
        // Initialize
        refreshCameras();
        refreshAlerts();
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            refreshCameras();
            refreshAlerts();
        }, 30000);
    </script>
</body>
</html>
    """
    return html_content


if __name__ == "__main__":
    print("🚀 Starting Project Argus API Gateway...")
    print("📍 API Gateway: http://localhost:8000")
    print("📊 Dashboard: http://localhost:8000/dashboard")
    print("📖 API Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "simple_api_gateway:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )