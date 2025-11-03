#!/usr/bin/env python3
"""
Demo Environment and Sample Data Creation for Project Argus
Creates demo environment with prerecorded video scenarios and sample crossing events.
"""

import pytest
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import cv2
from unittest.mock import Mock, patch

# Import core components
from shared.models.detection import Detection, DetectionResult, BoundingBox, DetectionClass
from shared.models.virtual_line import VirtualLine, Point
from shared.models.alerts import Alert, AlertType, Severity, CrossingEvent
from shared.models.incidents import Incident, IncidentStatus, Resolution, ResolutionType
from shared.models.evidence import Evidence, EvidenceType


class DemoEnvironmentBuilder:
    """Builder for creating comprehensive demo environment."""
    
    def __init__(self, demo_path):
        self.demo_path = Path(demo_path)
        self.demo_path.mkdir(parents=True, exist_ok=True)
        
        # Demo data storage
        self.cameras = []
        self.virtual_lines = []
        self.sample_detections = []
        self.sample_alerts = []
        self.sample_incidents = []
        self.sample_evidence = []
        self.video_scenarios = []
    
    def create_demo_cameras(self):
        """Create demo camera configurations."""
        camera_configs = [
            {
                "id": "demo_camera_001",
                "name": "Border Sector Alpha - Main Gate",
                "location": {"lat": 28.6139, "lng": 77.2090},
                "type": "visible_thermal",
                "resolution": "1920x1080",
                "fps": 30,
                "coverage_area": "Main border crossing point",
                "status": "active"
            },
            {
                "id": "demo_camera_002", 
                "name": "Border Sector Alpha - Perimeter East",
                "location": {"lat": 28.6140, "lng": 77.2095},
                "type": "thermal_night_vision",
                "resolution": "1280x720",
                "fps": 25,
                "coverage_area": "Eastern perimeter fence",
                "status": "active"
            },
            {
                "id": "demo_camera_003",
                "name": "Border Sector Beta - Patrol Route",
                "location": {"lat": 28.6135, "lng": 77.2085},
                "type": "visible_ptz",
                "resolution": "1920x1080", 
                "fps": 30,
                "coverage_area": "Mobile patrol route monitoring",
                "status": "active"
            },
            {
                "id": "demo_camera_004",
                "name": "Border Sector Beta - Remote Outpost",
                "location": {"lat": 28.6130, "lng": 77.2080},
                "type": "thermal_radar",
                "resolution": "640x480",
                "fps": 15,
                "coverage_area": "Remote area with radar integration",
                "status": "maintenance"
            }
        ]
        
        self.cameras = camera_configs
        
        # Save camera configurations
        cameras_file = self.demo_path / "cameras.json"
        with open(cameras_file, 'w') as f:
            json.dump(camera_configs, f, indent=2)
        
        return camera_configs
    
    def create_demo_virtual_lines(self):
        """Create demo virtual line configurations."""
        virtual_line_configs = [
            {
                "id": "vline_main_gate",
                "camera_id": "demo_camera_001",
                "name": "Main Gate Crossing Line",
                "points": [{"x": 400, "y": 200}, {"x": 400, "y": 600}],
                "type": "bidirectional",
                "sensitivity": 0.8,
                "active": True,
                "description": "Primary crossing detection at main gate"
            },
            {
                "id": "vline_perimeter_east",
                "camera_id": "demo_camera_002",
                "name": "Eastern Perimeter Breach Line",
                "points": [{"x": 100, "y": 300}, {"x": 700, "y": 350}],
                "type": "inbound_only",
                "sensitivity": 0.9,
                "active": True,
                "description": "Perimeter breach detection line"
            },
            {
                "id": "vline_patrol_route",
                "camera_id": "demo_camera_003",
                "name": "Patrol Route Monitoring",
                "points": [{"x": 300, "y": 100}, {"x": 500, "y": 500}],
                "type": "bidirectional",
                "sensitivity": 0.7,
                "active": True,
                "description": "Patrol route activity monitoring"
            }
        ]
        
        self.virtual_lines = virtual_line_configs
        
        # Save virtual line configurations
        vlines_file = self.demo_path / "virtual_lines.json"
        with open(vlines_file, 'w') as f:
            json.dump(virtual_line_configs, f, indent=2)
        
        return virtual_line_configs
    
    def create_sample_video_scenarios(self):
        """Create sample video scenarios for demonstration."""
        scenarios = [
            {
                "name": "authorized_crossing_daytime",
                "description": "Authorized personnel crossing during daytime",
                "camera_id": "demo_camera_001",
                "duration_seconds": 30,
                "scenario_type": "normal_operation",
                "expected_alerts": 0,
                "people_count": 1,
                "lighting": "daylight",
                "weather": "clear"
            },
            {
                "name": "unauthorized_crossing_night",
                "description": "Unauthorized crossing attempt at night",
                "camera_id": "demo_camera_002",
                "duration_seconds": 45,
                "scenario_type": "security_incident",
                "expected_alerts": 1,
                "people_count": 2,
                "lighting": "night",
                "weather": "clear"
            },
            {
                "name": "multiple_crossings_simultaneous",
                "description": "Multiple people crossing simultaneously",
                "camera_id": "demo_camera_001",
                "duration_seconds": 60,
                "scenario_type": "high_activity",
                "expected_alerts": 3,
                "people_count": 4,
                "lighting": "dawn",
                "weather": "foggy"
            },
            {
                "name": "false_positive_scenario",
                "description": "Animals or debris causing false detection",
                "camera_id": "demo_camera_003",
                "duration_seconds": 20,
                "scenario_type": "false_positive",
                "expected_alerts": 0,
                "people_count": 0,
                "lighting": "daylight",
                "weather": "windy"
            },
            {
                "name": "patrol_verification",
                "description": "Security patrol verification crossing",
                "camera_id": "demo_camera_003",
                "duration_seconds": 25,
                "scenario_type": "patrol_activity",
                "expected_alerts": 0,
                "people_count": 2,
                "lighting": "daylight",
                "weather": "clear"
            }
        ]
        
        # Generate actual video files for each scenario
        videos_dir = self.demo_path / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        for scenario in scenarios:
            video_path = self._generate_demo_video(scenario, videos_dir)
            scenario["video_path"] = str(video_path)
        
        self.video_scenarios = scenarios
        
        # Save scenario configurations
        scenarios_file = self.demo_path / "video_scenarios.json"
        with open(scenarios_file, 'w') as f:
            json.dump(scenarios, f, indent=2)
        
        return scenarios
    
    def _generate_demo_video(self, scenario, output_dir):
        """Generate a demo video file for a scenario."""
        video_path = output_dir / f"{scenario['name']}.mp4"
        
        # Video parameters
        width, height = 1280, 720
        fps = 25
        duration = scenario['duration_seconds']
        total_frames = fps * duration
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        try:
            for frame_num in range(total_frames):
                frame = self._generate_scenario_frame(scenario, frame_num, total_frames, width, height)
                out.write(frame)
        finally:
            out.release()
        
        return video_path
    
    def _generate_scenario_frame(self, scenario, frame_num, total_frames, width, height):
        """Generate a single frame for a video scenario."""
        # Create base frame based on lighting conditions
        if scenario['lighting'] == 'night':
            base_color = (20, 20, 40)  # Dark blue
        elif scenario['lighting'] == 'dawn':
            base_color = (60, 80, 120)  # Dawn colors
        else:
            base_color = (120, 150, 180)  # Daylight
        
        frame = np.full((height, width, 3), base_color, dtype=np.uint8)
        
        # Add weather effects
        if scenario['weather'] == 'foggy':
            # Add fog effect
            fog_overlay = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.7, fog_overlay, 0.3, 0)
        elif scenario['weather'] == 'windy':
            # Add slight motion blur
            kernel = np.ones((3, 3), np.float32) / 9
            frame = cv2.filter2D(frame, -1, kernel)
        
        # Add border/fence elements
        cv2.line(frame, (0, height//2), (width, height//2), (100, 100, 100), 3)  # Fence line
        cv2.rectangle(frame, (50, height//2-20), (150, height//2+20), (80, 80, 80), -1)  # Gate post
        cv2.rectangle(frame, (width-150, height//2-20), (width-50, height//2+20), (80, 80, 80), -1)  # Gate post
        
        # Add people based on scenario
        if scenario['people_count'] > 0:
            self._add_people_to_frame(frame, scenario, frame_num, total_frames, width, height)
        
        # Add timestamp
        timestamp = datetime.now() + timedelta(seconds=frame_num/25)
        cv2.putText(frame, timestamp.strftime("%Y-%m-%d %H:%M:%S"), 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add camera ID
        cv2.putText(frame, scenario['camera_id'], 
                   (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _add_people_to_frame(self, frame, scenario, frame_num, total_frames, width, height):
        """Add people to frame based on scenario."""
        people_count = scenario['people_count']
        
        for person_id in range(people_count):
            # Calculate person position based on frame progression
            progress = frame_num / total_frames
            
            if scenario['scenario_type'] == 'security_incident':
                # People moving stealthily from edge to center
                start_x = 50 + person_id * 30
                end_x = width - 100
                x = int(start_x + (end_x - start_x) * progress)
                y = height//2 + person_id * 20 - 10
            elif scenario['scenario_type'] == 'high_activity':
                # Multiple people crossing in different directions
                if person_id % 2 == 0:
                    x = int(100 + (width - 200) * progress)
                else:
                    x = int(width - 100 - (width - 200) * progress)
                y = height//2 + (person_id - people_count//2) * 40
            else:
                # Normal crossing pattern
                start_x = 100
                end_x = width - 100
                x = int(start_x + (end_x - start_x) * progress)
                y = height//2 + person_id * 30 - 15
            
            # Draw person (simple representation)
            person_color = (255, 255, 255) if scenario['lighting'] != 'night' else (180, 180, 180)
            
            # Body
            cv2.rectangle(frame, (x-10, y-30), (x+10, y+10), person_color, -1)
            # Head
            cv2.circle(frame, (x, y-40), 8, person_color, -1)
            # Legs
            cv2.line(frame, (x-5, y+10), (x-8, y+30), person_color, 3)
            cv2.line(frame, (x+5, y+10), (x+8, y+30), person_color, 3)
    
    def create_sample_detections(self):
        """Create sample detection data."""
        base_time = datetime.now() - timedelta(hours=24)
        
        sample_detections = []
        
        for i in range(50):
            detection_time = base_time + timedelta(minutes=i*15)
            camera_id = f"demo_camera_{(i % 3) + 1:03d}"
            
            detection = {
                "id": f"det_demo_{i+1:03d}",
                "camera_id": camera_id,
                "timestamp": detection_time.isoformat(),
                "bbox": {
                    "x": 200 + (i % 5) * 50,
                    "y": 150 + (i % 3) * 40,
                    "width": 40 + (i % 3) * 10,
                    "height": 80 + (i % 2) * 20
                },
                "confidence": 0.7 + (i % 3) * 0.1,
                "detection_class": "person",
                "features": [float(x) for x in np.random.rand(512)],
                "metadata": {
                    "lighting_condition": ["daylight", "dawn", "night"][i % 3],
                    "weather": ["clear", "foggy", "rainy"][i % 3]
                }
            }
            
            sample_detections.append(detection)
        
        self.sample_detections = sample_detections
        
        # Save detections
        detections_file = self.demo_path / "sample_detections.json"
        with open(detections_file, 'w') as f:
            json.dump(sample_detections, f, indent=2)
        
        return sample_detections
    
    def create_sample_alerts(self):
        """Create sample alert data."""
        base_time = datetime.now() - timedelta(hours=12)
        
        sample_alerts = []
        
        alert_scenarios = [
            {"type": "virtual_line_crossing", "severity": "high", "description": "Unauthorized crossing detected"},
            {"type": "virtual_line_crossing", "severity": "medium", "description": "Possible crossing - verification needed"},
            {"type": "tamper_detection", "severity": "critical", "description": "Camera tampering detected"},
            {"type": "system_health", "severity": "low", "description": "Camera offline - maintenance required"},
            {"type": "virtual_line_crossing", "severity": "high", "description": "Multiple simultaneous crossings"}
        ]
        
        for i in range(20):
            scenario = alert_scenarios[i % len(alert_scenarios)]
            alert_time = base_time + timedelta(minutes=i*30)
            
            alert = {
                "id": f"alert_demo_{i+1:03d}",
                "type": scenario["type"],
                "severity": scenario["severity"],
                "camera_id": f"demo_camera_{(i % 3) + 1:03d}",
                "detection_id": f"det_demo_{i+1:03d}",
                "timestamp": alert_time.isoformat(),
                "confidence": 0.8 + (i % 2) * 0.1,
                "risk_score": 0.6 + (i % 4) * 0.1,
                "description": scenario["description"],
                "acknowledged": i % 4 != 0,  # 75% acknowledged
                "acknowledged_by": f"operator_{(i % 3) + 1}" if i % 4 != 0 else None,
                "metadata": {
                    "crossing_direction": ["inbound", "outbound"][i % 2],
                    "virtual_line_id": f"vline_{['main_gate', 'perimeter_east', 'patrol_route'][i % 3]}"
                }
            }
            
            sample_alerts.append(alert)
        
        self.sample_alerts = sample_alerts
        
        # Save alerts
        alerts_file = self.demo_path / "sample_alerts.json"
        with open(alerts_file, 'w') as f:
            json.dump(sample_alerts, f, indent=2)
        
        return sample_alerts
    
    def create_sample_incidents(self):
        """Create sample incident data."""
        base_time = datetime.now() - timedelta(hours=6)
        
        sample_incidents = []
        
        incident_types = [
            {"status": "resolved", "resolution": "confirmed_crossing", "description": "Unauthorized crossing confirmed - patrol dispatched"},
            {"status": "open", "resolution": None, "description": "Investigation ongoing - reviewing footage"},
            {"status": "resolved", "resolution": "false_positive", "description": "False positive - animal movement"},
            {"status": "escalated", "resolution": None, "description": "Multiple crossings - supervisor notified"},
            {"status": "closed", "resolution": "confirmed_crossing", "description": "Incident resolved - subjects apprehended"}
        ]
        
        for i in range(10):
            incident_type = incident_types[i % len(incident_types)]
            incident_time = base_time + timedelta(hours=i)
            
            incident = {
                "id": f"inc_demo_{i+1:03d}",
                "alert_id": f"alert_demo_{i+1:03d}",
                "status": incident_type["status"],
                "priority": ["low", "medium", "high", "critical"][i % 4],
                "created_at": incident_time.isoformat(),
                "created_by": f"operator_{(i % 3) + 1}",
                "assigned_to": f"operator_{(i % 3) + 1}",
                "description": incident_type["description"],
                "resolution": incident_type["resolution"],
                "evidence_ids": [f"evidence_demo_{i+1:03d}", f"evidence_demo_{i+10:03d}"],
                "notes": [
                    {
                        "timestamp": (incident_time + timedelta(minutes=5)).isoformat(),
                        "author": f"operator_{(i % 3) + 1}",
                        "content": "Initial assessment completed",
                        "type": "investigation"
                    },
                    {
                        "timestamp": (incident_time + timedelta(minutes=15)).isoformat(),
                        "author": f"operator_{(i % 3) + 1}",
                        "content": "Additional evidence collected",
                        "type": "evidence"
                    }
                ],
                "metadata": {
                    "location": f"Border Sector {['Alpha', 'Beta', 'Gamma'][i % 3]}",
                    "weather_conditions": ["clear", "foggy", "rainy"][i % 3],
                    "patrol_notified": i % 2 == 0
                }
            }
            
            sample_incidents.append(incident)
        
        self.sample_incidents = sample_incidents
        
        # Save incidents
        incidents_file = self.demo_path / "sample_incidents.json"
        with open(incidents_file, 'w') as f:
            json.dump(sample_incidents, f, indent=2)
        
        return sample_incidents
    
    def create_sample_evidence(self):
        """Create sample evidence metadata."""
        base_time = datetime.now() - timedelta(hours=3)
        
        sample_evidence = []
        
        evidence_types = ["image", "video", "metadata", "audio"]
        
        for i in range(30):
            evidence_time = base_time + timedelta(minutes=i*10)
            
            evidence = {
                "id": f"evidence_demo_{i+1:03d}",
                "type": evidence_types[i % len(evidence_types)],
                "incident_id": f"inc_demo_{(i % 10) + 1:03d}",
                "camera_id": f"demo_camera_{(i % 3) + 1:03d}",
                "created_at": evidence_time.isoformat(),
                "created_by": f"operator_{(i % 3) + 1}",
                "file_path": f"evidence/demo_{i+1:03d}.{['jpg', 'mp4', 'json', 'wav'][i % 4]}",
                "file_size": 1024 * (100 + i * 50),  # Varying file sizes
                "hash_sha256": f"demo_hash_{i+1:032d}",
                "hmac_signature": f"demo_signature_{i+1:064d}",
                "status": ["sealed", "pending", "verified"][i % 3],
                "metadata": {
                    "original_filename": f"camera_capture_{evidence_time.strftime('%Y%m%d_%H%M%S')}.jpg",
                    "resolution": "1920x1080" if i % 2 == 0 else "1280x720",
                    "duration": f"{10 + i % 30}s" if evidence_types[i % len(evidence_types)] == "video" else None,
                    "confidence": 0.8 + (i % 2) * 0.1
                },
                "chain_of_custody": [
                    {
                        "timestamp": evidence_time.isoformat(),
                        "action": "created",
                        "user_id": f"operator_{(i % 3) + 1}",
                        "notes": "Evidence automatically captured"
                    },
                    {
                        "timestamp": (evidence_time + timedelta(minutes=5)).isoformat(),
                        "action": "sealed",
                        "user_id": f"supervisor_{(i % 2) + 1}",
                        "notes": "Evidence sealed for investigation"
                    }
                ]
            }
            
            sample_evidence.append(evidence)
        
        self.sample_evidence = sample_evidence
        
        # Save evidence
        evidence_file = self.demo_path / "sample_evidence.json"
        with open(evidence_file, 'w') as f:
            json.dump(sample_evidence, f, indent=2)
        
        return sample_evidence
    
    def create_demo_dashboard_config(self):
        """Create demo dashboard configuration."""
        dashboard_config = {
            "title": "Project Argus - Border Security Demo",
            "version": "1.0.0",
            "demo_mode": True,
            "auto_refresh_interval": 5000,  # 5 seconds
            "map_config": {
                "center": {"lat": 28.6139, "lng": 77.2090},
                "zoom": 15,
                "style": "satellite"
            },
            "camera_grid": {
                "layout": "2x2",
                "auto_cycle": True,
                "cycle_interval": 10000  # 10 seconds
            },
            "alert_settings": {
                "sound_enabled": True,
                "popup_enabled": True,
                "auto_acknowledge_timeout": 300000  # 5 minutes
            },
            "demo_scenarios": [
                {
                    "name": "Normal Operations",
                    "description": "Typical day with authorized crossings",
                    "duration": "5 minutes",
                    "cameras": ["demo_camera_001", "demo_camera_003"]
                },
                {
                    "name": "Security Incident",
                    "description": "Unauthorized crossing with multiple alerts",
                    "duration": "3 minutes", 
                    "cameras": ["demo_camera_002"]
                },
                {
                    "name": "System Maintenance",
                    "description": "Camera maintenance and health monitoring",
                    "duration": "2 minutes",
                    "cameras": ["demo_camera_004"]
                }
            ]
        }
        
        # Save dashboard config
        dashboard_file = self.demo_path / "dashboard_config.json"
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_config, f, indent=2)
        
        return dashboard_config
    
    def create_demo_readme(self):
        """Create README file for demo environment."""
        readme_content = """# Project Argus Demo Environment

This demo environment showcases the capabilities of Project Argus border security system.

## Demo Components

### 1. Camera Feeds
- **demo_camera_001**: Main Gate - Primary crossing point monitoring
- **demo_camera_002**: Perimeter East - Thermal night vision coverage  
- **demo_camera_003**: Patrol Route - Mobile patrol monitoring
- **demo_camera_004**: Remote Outpost - Radar-integrated monitoring

### 2. Video Scenarios
- **authorized_crossing_daytime**: Normal authorized personnel crossing
- **unauthorized_crossing_night**: Security incident with night vision
- **multiple_crossings_simultaneous**: High activity scenario
- **false_positive_scenario**: Environmental false positive handling
- **patrol_verification**: Security patrol verification

### 3. Sample Data
- **50 sample detections** across 24-hour period
- **20 sample alerts** with various severity levels
- **10 sample incidents** showing complete workflow
- **30 evidence items** with chain of custody

### 4. Interactive Features
- Real-time alert dashboard
- Incident management workflow
- Evidence review and export
- System health monitoring
- Performance analytics

## Running the Demo

1. **Start Demo Environment**:
   ```bash
   python tests/integration/run_demo.py
   ```

2. **Access Dashboard**:
   - Open browser to http://localhost:3000
   - Login with demo credentials (operator/demo123)

3. **Demo Scenarios**:
   - Use scenario selector to run different demonstrations
   - Monitor alerts and incidents in real-time
   - Review evidence and generate reports

## Demo Credentials

- **Operator**: username=`demo_operator`, password=`demo123`
- **Supervisor**: username=`demo_supervisor`, password=`demo456`
- **Auditor**: username=`demo_auditor`, password=`demo789`

## Key Demonstrations

### Detection and Tracking
- Person detection with confidence scoring
- Multi-camera tracking and re-identification
- Virtual line crossing detection

### Alert Management
- Real-time alert generation
- Risk scoring and prioritization
- Escalation workflows

### Incident Response
- Incident creation and assignment
- Evidence collection and sealing
- Resolution and reporting

### Security Features
- Encrypted evidence storage
- Audit trail maintenance
- Role-based access control

### System Monitoring
- Camera health monitoring
- Performance metrics
- Tamper detection

## Technical Specifications

- **Detection Latency**: < 300ms average
- **False Positive Rate**: < 1 alert/camera/day
- **System Uptime**: 99.9% target
- **Concurrent Cameras**: 100+ supported
- **Evidence Encryption**: AES-256
- **Data Integrity**: HMAC-SHA256

## Support

For technical support or questions about the demo:
- Email: support@projectargus.demo
- Documentation: /docs/
- API Reference: /api/docs/

---
*Project Argus Demo Environment v1.0*
*Generated: {timestamp}*
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        readme_file = self.demo_path / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        return readme_file
    
    def build_complete_demo(self):
        """Build complete demo environment with all components."""
        print("🏗️  Building Project Argus Demo Environment...")
        
        # Create all demo components
        cameras = self.create_demo_cameras()
        print(f"✅ Created {len(cameras)} demo cameras")
        
        virtual_lines = self.create_demo_virtual_lines()
        print(f"✅ Created {len(virtual_lines)} virtual lines")
        
        video_scenarios = self.create_sample_video_scenarios()
        print(f"✅ Generated {len(video_scenarios)} video scenarios")
        
        detections = self.create_sample_detections()
        print(f"✅ Created {len(detections)} sample detections")
        
        alerts = self.create_sample_alerts()
        print(f"✅ Created {len(alerts)} sample alerts")
        
        incidents = self.create_sample_incidents()
        print(f"✅ Created {len(incidents)} sample incidents")
        
        evidence = self.create_sample_evidence()
        print(f"✅ Created {len(evidence)} evidence items")
        
        dashboard_config = self.create_demo_dashboard_config()
        print("✅ Created dashboard configuration")
        
        readme_file = self.create_demo_readme()
        print(f"✅ Created README: {readme_file}")
        
        # Create summary
        summary = {
            "demo_path": str(self.demo_path),
            "created_at": datetime.now().isoformat(),
            "components": {
                "cameras": len(cameras),
                "virtual_lines": len(virtual_lines),
                "video_scenarios": len(video_scenarios),
                "detections": len(detections),
                "alerts": len(alerts),
                "incidents": len(incidents),
                "evidence": len(evidence)
            },
            "total_files": len(list(self.demo_path.rglob("*"))),
            "demo_ready": True
        }
        
        summary_file = self.demo_path / "demo_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"🎉 Demo environment ready at: {self.demo_path}")
        print(f"📊 Summary saved to: {summary_file}")
        
        return summary


class TestDemoEnvironment:
    """Test demo environment creation and functionality."""
    
    def setup_method(self):
        """Set up demo test fixtures."""
        self.temp_demo_dir = tempfile.mkdtemp(prefix="argus_demo_test_")
        self.demo_builder = DemoEnvironmentBuilder(self.temp_demo_dir)
    
    def teardown_method(self):
        """Clean up demo test fixtures."""
        shutil.rmtree(self.temp_demo_dir, ignore_errors=True)
    
    def test_demo_camera_creation(self):
        """Test demo camera configuration creation."""
        cameras = self.demo_builder.create_demo_cameras()
        
        # Verify cameras were created
        assert len(cameras) == 4
        
        # Verify camera structure
        for camera in cameras:
            assert "id" in camera
            assert "name" in camera
            assert "location" in camera
            assert "type" in camera
            assert "resolution" in camera
            assert "fps" in camera
            assert "status" in camera
            
            # Verify location has lat/lng
            assert "lat" in camera["location"]
            assert "lng" in camera["location"]
        
        # Verify cameras file was created
        cameras_file = Path(self.temp_demo_dir) / "cameras.json"
        assert cameras_file.exists()
        
        # Verify file content
        with open(cameras_file, 'r') as f:
            saved_cameras = json.load(f)
        assert len(saved_cameras) == 4
        assert saved_cameras[0]["id"] == "demo_camera_001"
    
    def test_demo_virtual_lines_creation(self):
        """Test demo virtual line configuration creation."""
        virtual_lines = self.demo_builder.create_demo_virtual_lines()
        
        # Verify virtual lines were created
        assert len(virtual_lines) == 3
        
        # Verify virtual line structure
        for vline in virtual_lines:
            assert "id" in vline
            assert "camera_id" in vline
            assert "name" in vline
            assert "points" in vline
            assert "type" in vline
            assert "sensitivity" in vline
            assert "active" in vline
            
            # Verify points structure
            assert len(vline["points"]) >= 2
            for point in vline["points"]:
                assert "x" in point
                assert "y" in point
        
        # Verify virtual lines file was created
        vlines_file = Path(self.temp_demo_dir) / "virtual_lines.json"
        assert vlines_file.exists()
    
    def test_sample_video_generation(self):
        """Test sample video scenario generation."""
        scenarios = self.demo_builder.create_sample_video_scenarios()
        
        # Verify scenarios were created
        assert len(scenarios) >= 3
        
        # Verify scenario structure
        for scenario in scenarios:
            assert "name" in scenario
            assert "description" in scenario
            assert "camera_id" in scenario
            assert "duration_seconds" in scenario
            assert "scenario_type" in scenario
            assert "expected_alerts" in scenario
            assert "people_count" in scenario
            assert "lighting" in scenario
            assert "weather" in scenario
            assert "video_path" in scenario
            
            # Verify video file was created
            video_path = Path(scenario["video_path"])
            assert video_path.exists()
            assert video_path.suffix == ".mp4"
            assert video_path.stat().st_size > 0
    
    def test_sample_data_creation(self):
        """Test sample detection, alert, and incident data creation."""
        # Create sample detections
        detections = self.demo_builder.create_sample_detections()
        assert len(detections) == 50
        
        # Verify detection structure
        detection = detections[0]
        assert "id" in detection
        assert "camera_id" in detection
        assert "timestamp" in detection
        assert "bbox" in detection
        assert "confidence" in detection
        assert "detection_class" in detection
        
        # Create sample alerts
        alerts = self.demo_builder.create_sample_alerts()
        assert len(alerts) == 20
        
        # Verify alert structure
        alert = alerts[0]
        assert "id" in alert
        assert "type" in alert
        assert "severity" in alert
        assert "camera_id" in alert
        assert "timestamp" in alert
        
        # Create sample incidents
        incidents = self.demo_builder.create_sample_incidents()
        assert len(incidents) == 10
        
        # Verify incident structure
        incident = incidents[0]
        assert "id" in incident
        assert "alert_id" in incident
        assert "status" in incident
        assert "priority" in incident
        assert "created_at" in incident
        assert "created_by" in incident
    
    def test_sample_evidence_creation(self):
        """Test sample evidence data creation."""
        evidence_items = self.demo_builder.create_sample_evidence()
        
        # Verify evidence was created
        assert len(evidence_items) == 30
        
        # Verify evidence structure
        evidence = evidence_items[0]
        assert "id" in evidence
        assert "type" in evidence
        assert "incident_id" in evidence
        assert "camera_id" in evidence
        assert "created_at" in evidence
        assert "file_path" in evidence
        assert "hash_sha256" in evidence
        assert "hmac_signature" in evidence
        assert "chain_of_custody" in evidence
        
        # Verify chain of custody structure
        custody = evidence["chain_of_custody"]
        assert len(custody) >= 1
        assert "timestamp" in custody[0]
        assert "action" in custody[0]
        assert "user_id" in custody[0]
    
    def test_dashboard_config_creation(self):
        """Test dashboard configuration creation."""
        config = self.demo_builder.create_demo_dashboard_config()
        
        # Verify config structure
        assert "title" in config
        assert "demo_mode" in config
        assert config["demo_mode"] is True
        assert "map_config" in config
        assert "camera_grid" in config
        assert "alert_settings" in config
        assert "demo_scenarios" in config
        
        # Verify map config
        map_config = config["map_config"]
        assert "center" in map_config
        assert "lat" in map_config["center"]
        assert "lng" in map_config["center"]
        
        # Verify demo scenarios
        scenarios = config["demo_scenarios"]
        assert len(scenarios) >= 3
        for scenario in scenarios:
            assert "name" in scenario
            assert "description" in scenario
            assert "duration" in scenario
            assert "cameras" in scenario
    
    def test_complete_demo_build(self):
        """Test complete demo environment build."""
        summary = self.demo_builder.build_complete_demo()
        
        # Verify summary structure
        assert "demo_path" in summary
        assert "created_at" in summary
        assert "components" in summary
        assert "total_files" in summary
        assert "demo_ready" in summary
        assert summary["demo_ready"] is True
        
        # Verify components counts
        components = summary["components"]
        assert components["cameras"] == 4
        assert components["virtual_lines"] == 3
        assert components["video_scenarios"] >= 3
        assert components["detections"] == 50
        assert components["alerts"] == 20
        assert components["incidents"] == 10
        assert components["evidence"] == 30
        
        # Verify files were created
        demo_path = Path(summary["demo_path"])
        assert (demo_path / "cameras.json").exists()
        assert (demo_path / "virtual_lines.json").exists()
        assert (demo_path / "video_scenarios.json").exists()
        assert (demo_path / "sample_detections.json").exists()
        assert (demo_path / "sample_alerts.json").exists()
        assert (demo_path / "sample_incidents.json").exists()
        assert (demo_path / "sample_evidence.json").exists()
        assert (demo_path / "dashboard_config.json").exists()
        assert (demo_path / "README.md").exists()
        assert (demo_path / "demo_summary.json").exists()
        
        # Verify videos directory exists
        videos_dir = demo_path / "videos"
        assert videos_dir.exists()
        video_files = list(videos_dir.glob("*.mp4"))
        assert len(video_files) >= 3
    
    def test_demo_data_integrity(self):
        """Test integrity of generated demo data."""
        # Build complete demo
        summary = self.demo_builder.build_complete_demo()
        demo_path = Path(summary["demo_path"])
        
        # Load and verify cameras
        with open(demo_path / "cameras.json", 'r') as f:
            cameras = json.load(f)
        
        camera_ids = [cam["id"] for cam in cameras]
        
        # Load and verify virtual lines reference valid cameras
        with open(demo_path / "virtual_lines.json", 'r') as f:
            virtual_lines = json.load(f)
        
        for vline in virtual_lines:
            assert vline["camera_id"] in camera_ids
        
        # Load and verify detections reference valid cameras
        with open(demo_path / "sample_detections.json", 'r') as f:
            detections = json.load(f)
        
        for detection in detections[:5]:  # Check first 5
            # Camera ID should be in format demo_camera_XXX
            assert detection["camera_id"].startswith("demo_camera_")
            
            # Timestamp should be valid ISO format
            datetime.fromisoformat(detection["timestamp"])
            
            # Confidence should be valid range
            assert 0.0 <= detection["confidence"] <= 1.0
        
        # Load and verify alerts reference valid detections
        with open(demo_path / "sample_alerts.json", 'r') as f:
            alerts = json.load(f)
        
        detection_ids = [det["id"] for det in detections]
        
        for alert in alerts[:5]:  # Check first 5
            assert alert["detection_id"] in detection_ids
            assert alert["camera_id"].startswith("demo_camera_")
            
            # Severity should be valid
            assert alert["severity"] in ["low", "medium", "high", "critical"]
            
            # Risk score should be valid range
            assert 0.0 <= alert["risk_score"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])