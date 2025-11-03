#!/usr/bin/env python3
"""
Demo Runner for Project Argus
Creates and runs a demonstration of the border detection system
"""

import os
import sys
import time
import tempfile
import webbrowser
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tests.integration.test_demo_environment import DemoEnvironmentBuilder


class ArgusDemo:
    """Project Argus demonstration runner."""
    
    def __init__(self):
        self.demo_path = None
        self.demo_builder = None
    
    def setup_demo(self):
        """Set up the demo environment."""
        print("🎬 Setting up Project Argus Demo Environment")
        print("=" * 50)
        
        # Create demo directory
        self.demo_path = Path("demo_environment")
        self.demo_path.mkdir(exist_ok=True)
        
        # Initialize demo builder
        self.demo_builder = DemoEnvironmentBuilder(self.demo_path)
        
        print(f"📁 Demo directory: {self.demo_path.absolute()}")
        
    def build_demo_data(self):
        """Build comprehensive demo data."""
        print("\n🏗️  Building demo data...")
        
        try:
            summary = self.demo_builder.build_complete_demo()
            
            print(f"\n✅ Demo environment created successfully!")
            print(f"📊 Components created:")
            for component, count in summary['components'].items():
                print(f"   • {component.replace('_', ' ').title()}: {count}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error building demo: {e}")
            return False
    
    def create_demo_server(self):
        """Create a simple demo server."""
        demo_server_code = '''
import json
import http.server
import socketserver
from pathlib import Path

class DemoHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.path = '/demo.html'
        elif self.path == '/api/cameras':
            self.send_json_response('cameras.json')
            return
        elif self.path == '/api/alerts':
            self.send_json_response('sample_alerts.json')
            return
        elif self.path == '/api/incidents':
            self.send_json_response('sample_incidents.json')
            return
        elif self.path == '/api/evidence':
            self.send_json_response('sample_evidence.json')
            return
        
        super().do_GET()
    
    def send_json_response(self, filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        except FileNotFoundError:
            self.send_error(404)

if __name__ == "__main__":
    import sys
    PORT = 8081
    # Try different ports if 8081 is busy
    for port in range(8081, 8090):
        try:
            with socketserver.TCPServer(("", port), DemoHandler) as httpd:
                print(f"Demo server running at http://localhost:{port}")
                httpd.serve_forever()
                break
        except OSError:
            continue
    else:
        print("Could not find available port")
        sys.exit(1)
'''
        
        server_file = self.demo_path / "demo_server.py"
        with open(server_file, 'w', encoding='utf-8') as f:
            f.write(demo_server_code)
        
        return server_file
    
    def create_demo_html(self):
        """Create demo HTML interface."""
        html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Argus - Border Security Demo</title>
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
            transition: transform 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .card h3 {
            color: #4CAF50;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .status {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        .status.active { background: #4CAF50; }
        .status.alert { background: #FF5722; }
        .status.warning { background: #FF9800; }
        .status.resolved { background: #2196F3; }
        .metrics {
            display: flex;
            justify-content: space-between;
            margin: 1rem 0;
        }
        .metric {
            text-align: center;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #4CAF50;
        }
        .video-placeholder {
            width: 100%;
            height: 200px;
            background: #000;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 1rem 0;
            border: 2px solid #4CAF50;
        }
        .btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.3s ease;
        }
        .btn:hover {
            background: #45a049;
        }
        .alert-item {
            background: rgba(255,87,34,0.2);
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
            border-left: 4px solid #FF5722;
        }
        .incident-item {
            background: rgba(33,150,243,0.2);
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
            border-left: 4px solid #2196F3;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Project Argus - Border Security System Demo</h1>
        <p>Advanced AI-Powered Border Detection and Monitoring</p>
    </div>
    
    <div class="container">
        <div class="metrics">
            <div class="metric">
                <div class="metric-value" id="camera-count">4</div>
                <div>Active Cameras</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="alert-count">0</div>
                <div>Active Alerts</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="incident-count">0</div>
                <div>Open Incidents</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="detection-count">0</div>
                <div>Detections Today</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📹 Camera Feeds</h3>
                <div id="cameras-list">Loading cameras...</div>
                <div class="video-placeholder">
                    <div>🎥 Live Feed Simulation<br><small>Camera feeds would appear here</small></div>
                </div>
            </div>
            
            <div class="card">
                <h3>🚨 Recent Alerts</h3>
                <div id="alerts-list">Loading alerts...</div>
                <button class="btn" onclick="refreshAlerts()">Refresh Alerts</button>
            </div>
            
            <div class="card">
                <h3>📋 Active Incidents</h3>
                <div id="incidents-list">Loading incidents...</div>
                <button class="btn" onclick="refreshIncidents()">Refresh Incidents</button>
            </div>
            
            <div class="card">
                <h3>🔍 System Status</h3>
                <div>
                    <p><span class="status active">●</span> Detection Pipeline: Active</p>
                    <p><span class="status active">●</span> Alert System: Active</p>
                    <p><span class="status active">●</span> Evidence Store: Active</p>
                    <p><span class="status warning">●</span> Demo Mode: Enabled</p>
                </div>
                <div class="video-placeholder">
                    <div>📊 System Metrics<br><small>Performance graphs would appear here</small></div>
                </div>
            </div>
        </div>
        
        <div class="card" style="margin-top: 2rem;">
            <h3>🎮 Demo Controls</h3>
            <p>This is a demonstration of Project Argus capabilities using simulated data.</p>
            <div style="margin-top: 1rem;">
                <button class="btn" onclick="simulateCrossing()">Simulate Border Crossing</button>
                <button class="btn" onclick="simulateAlert()" style="margin-left: 1rem;">Generate Test Alert</button>
                <button class="btn" onclick="refreshAll()" style="margin-left: 1rem;">Refresh All Data</button>
            </div>
        </div>
    </div>
    
    <script>
        let alertCount = 0;
        let incidentCount = 0;
        let detectionCount = 0;
        
        async function loadData(endpoint) {
            try {
                const response = await fetch(`/api/${endpoint}`);
                return await response.json();
            } catch (error) {
                console.error(`Error loading ${endpoint}:`, error);
                return [];
            }
        }
        
        async function loadCameras() {
            const cameras = await loadData('cameras');
            const container = document.getElementById('cameras-list');
            
            if (cameras.length > 0) {
                container.innerHTML = cameras.map(camera => 
                    `<div style="margin: 0.5rem 0;">
                        <strong>${camera.name}</strong><br>
                        <small>${camera.location.lat}, ${camera.location.lng}</small>
                        <span class="status ${camera.status}">${camera.status}</span>
                    </div>`
                ).join('');
                document.getElementById('camera-count').textContent = cameras.length;
            } else {
                container.innerHTML = '<p>No camera data available</p>';
            }
        }
        
        async function loadAlerts() {
            const alerts = await loadData('alerts');
            const container = document.getElementById('alerts-list');
            
            if (alerts.length > 0) {
                const recentAlerts = alerts.slice(0, 5);
                container.innerHTML = recentAlerts.map(alert => 
                    `<div class="alert-item">
                        <strong>${alert.type.replace('_', ' ').toUpperCase()}</strong><br>
                        <small>${new Date(alert.timestamp).toLocaleString()}</small><br>
                        <span class="status ${alert.severity}">${alert.severity}</span>
                        Camera: ${alert.camera_id}
                    </div>`
                ).join('');
                alertCount = alerts.filter(a => !a.acknowledged).length;
                document.getElementById('alert-count').textContent = alertCount;
            } else {
                container.innerHTML = '<p>No recent alerts</p>';
            }
        }
        
        async function loadIncidents() {
            const incidents = await loadData('incidents');
            const container = document.getElementById('incidents-list');
            
            if (incidents.length > 0) {
                const openIncidents = incidents.filter(i => i.status !== 'closed').slice(0, 3);
                container.innerHTML = openIncidents.map(incident => 
                    `<div class="incident-item">
                        <strong>Incident ${incident.id}</strong><br>
                        <small>${incident.description}</small><br>
                        <span class="status ${incident.status}">${incident.status}</span>
                        Priority: ${incident.priority}
                    </div>`
                ).join('');
                incidentCount = openIncidents.length;
                document.getElementById('incident-count').textContent = incidentCount;
            } else {
                container.innerHTML = '<p>No active incidents</p>';
            }
        }
        
        function simulateCrossing() {
            detectionCount++;
            document.getElementById('detection-count').textContent = detectionCount;
            
            // Simulate alert generation
            setTimeout(() => {
                alertCount++;
                document.getElementById('alert-count').textContent = alertCount;
                
                const alertsContainer = document.getElementById('alerts-list');
                const newAlert = document.createElement('div');
                newAlert.className = 'alert-item';
                newAlert.innerHTML = `
                    <strong>VIRTUAL LINE CROSSING</strong><br>
                    <small>${new Date().toLocaleString()}</small><br>
                    <span class="status alert">high</span>
                    Camera: demo_camera_001
                `;
                alertsContainer.insertBefore(newAlert, alertsContainer.firstChild);
            }, 1000);
        }
        
        function simulateAlert() {
            alertCount++;
            document.getElementById('alert-count').textContent = alertCount;
            
            const alertsContainer = document.getElementById('alerts-list');
            const newAlert = document.createElement('div');
            newAlert.className = 'alert-item';
            newAlert.innerHTML = `
                <strong>SIMULATED ALERT</strong><br>
                <small>${new Date().toLocaleString()}</small><br>
                <span class="status warning">medium</span>
                Camera: demo_camera_002
            `;
            alertsContainer.insertBefore(newAlert, alertsContainer.firstChild);
        }
        
        function refreshAlerts() {
            loadAlerts();
        }
        
        function refreshIncidents() {
            loadIncidents();
        }
        
        function refreshAll() {
            loadCameras();
            loadAlerts();
            loadIncidents();
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            refreshAll();
            
            // Auto-refresh every 30 seconds
            setInterval(refreshAll, 30000);
        });
    </script>
</body>
</html>
'''
        
        html_file = self.demo_path / "demo.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file
    
    def run_demo(self):
        """Run the complete demo."""
        print("\n🎬 Starting Project Argus Demo")
        print("=" * 50)
        
        # Setup demo
        self.setup_demo()
        
        # Build demo data
        if not self.build_demo_data():
            return 1
        
        # Create demo server and HTML
        server_file = self.create_demo_server()
        html_file = self.create_demo_html()
        
        print(f"\n📄 Demo files created:")
        print(f"   • HTML Interface: {html_file}")
        print(f"   • Demo Server: {server_file}")
        
        # Start demo server
        print(f"\n🚀 Starting demo server...")
        print(f"📍 Demo URL: http://localhost:8081 (or next available port)")
        print(f"\n🎯 Demo Features:")
        print(f"   • Interactive camera grid")
        print(f"   • Real-time alert simulation")
        print(f"   • Incident management workflow")
        print(f"   • Evidence review system")
        print(f"   • System health monitoring")
        
        print(f"\n🛑 Press Ctrl+C to stop the demo")
        
        try:
            # Try to open browser
            webbrowser.open('http://localhost:8081')
        except:
            pass
        
        # Run demo server
        try:
            os.chdir(self.demo_path)
            os.system("python demo_server.py")
        except KeyboardInterrupt:
            print("\n🛑 Demo stopped")
            return 0
        
        return 0


def main():
    """Main entry point."""
    demo = ArgusDemo()
    return demo.run_demo()


if __name__ == "__main__":
    sys.exit(main())