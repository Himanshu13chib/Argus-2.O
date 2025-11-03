#!/usr/bin/env python3
"""
Quick Demo for Project Argus - Opens immediately
"""

import webbrowser
import http.server
import socketserver
import threading
import time
import json
from pathlib import Path


def create_quick_demo_html():
    """Create a quick demo HTML page."""
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Argus - Border Security System</title>
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
            padding: 2rem;
            text-align: center;
            border-bottom: 3px solid #4CAF50;
        }
        .header h1 {
            font-size: 3rem;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 2rem;
            margin-top: 2rem;
        }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: all 0.3s ease;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .card:hover {
            transform: translateY(-10px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        }
        .card h3 {
            color: #4CAF50;
            margin-bottom: 1rem;
            font-size: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .status {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-size: 0.9rem;
            font-weight: bold;
            margin: 0.25rem;
        }
        .status.active { background: #4CAF50; }
        .status.alert { background: #FF5722; animation: pulse 2s infinite; }
        .status.warning { background: #FF9800; }
        .status.resolved { background: #2196F3; }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }
        .metric {
            text-align: center;
            background: rgba(255,255,255,0.1);
            padding: 1.5rem;
            border-radius: 10px;
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #4CAF50;
            display: block;
        }
        .video-placeholder {
            width: 100%;
            height: 200px;
            background: linear-gradient(45deg, #000 25%, #333 25%, #333 50%, #000 50%, #000 75%, #333 75%);
            background-size: 20px 20px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 1rem 0;
            border: 2px solid #4CAF50;
            position: relative;
            overflow: hidden;
        }
        .video-placeholder::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(76, 175, 80, 0.3), transparent);
            animation: scan 3s infinite;
        }
        @keyframes scan {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        .btn {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: bold;
            transition: all 0.3s ease;
            margin: 0.5rem;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
        }
        .alert-item, .incident-item {
            background: rgba(255,87,34,0.2);
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid #FF5722;
            transition: all 0.3s ease;
        }
        .incident-item {
            background: rgba(33,150,243,0.2);
            border-left-color: #2196F3;
        }
        .alert-item:hover, .incident-item:hover {
            transform: translateX(5px);
        }
        .feature-list {
            list-style: none;
            padding: 0;
        }
        .feature-list li {
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .feature-list li:before {
            content: '✓';
            color: #4CAF50;
            font-weight: bold;
            margin-right: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Project Argus</h1>
        <p style="font-size: 1.2rem;">Advanced AI-Powered Border Detection and Monitoring System</p>
        <p style="margin-top: 1rem; opacity: 0.8;">Real-time threat detection • Multi-camera tracking • Intelligent alerts</p>
    </div>
    
    <div class="container">
        <div class="metrics">
            <div class="metric">
                <span class="metric-value" id="camera-count">4</span>
                <div>Active Cameras</div>
            </div>
            <div class="metric">
                <span class="metric-value" id="alert-count">2</span>
                <div>Active Alerts</div>
            </div>
            <div class="metric">
                <span class="metric-value" id="incident-count">1</span>
                <div>Open Incidents</div>
            </div>
            <div class="metric">
                <span class="metric-value" id="detection-count">47</span>
                <div>Detections Today</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📹 Live Camera Feeds</h3>
                <div>
                    <p><span class="status active">●</span> Border Sector Alpha - Main Gate</p>
                    <p><span class="status active">●</span> Border Sector Alpha - Perimeter East</p>
                    <p><span class="status active">●</span> Border Sector Beta - Patrol Route</p>
                    <p><span class="status warning">●</span> Border Sector Beta - Remote Outpost</p>
                </div>
                <div class="video-placeholder">
                    <div style="text-align: center; z-index: 1;">
                        🎥 Live Feed Simulation<br>
                        <small>AI Detection Active</small>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>🚨 Recent Alerts</h3>
                <div class="alert-item">
                    <strong>VIRTUAL LINE CROSSING</strong><br>
                    <small>2 minutes ago • Camera: Border Alpha Main</small><br>
                    <span class="status alert">HIGH</span>
                    Confidence: 94%
                </div>
                <div class="alert-item">
                    <strong>MOTION DETECTED</strong><br>
                    <small>15 minutes ago • Camera: Perimeter East</small><br>
                    <span class="status warning">MEDIUM</span>
                    Confidence: 78%
                </div>
                <button class="btn" onclick="simulateAlert()">Generate Test Alert</button>
            </div>
            
            <div class="card">
                <h3>📋 Active Incidents</h3>
                <div class="incident-item">
                    <strong>Incident #INC-2024-001</strong><br>
                    <small>Unauthorized crossing detected - Investigation ongoing</small><br>
                    <span class="status alert">OPEN</span>
                    Priority: High
                </div>
                <button class="btn" onclick="simulateIncident()">Create Test Incident</button>
            </div>
            
            <div class="card">
                <h3>🔍 System Status</h3>
                <div>
                    <p><span class="status active">●</span> AI Detection Pipeline: Active</p>
                    <p><span class="status active">●</span> Multi-Camera Tracking: Active</p>
                    <p><span class="status active">●</span> Alert System: Active</p>
                    <p><span class="status active">●</span> Evidence Store: Active</p>
                    <p><span class="status warning">●</span> Demo Mode: Enabled</p>
                </div>
                <div class="video-placeholder">
                    <div style="text-align: center; z-index: 1;">
                        📊 System Performance<br>
                        <small>Latency: 127ms • Uptime: 99.9%</small>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>🎯 Key Features</h3>
                <ul class="feature-list">
                    <li>Real-time person detection with 95%+ accuracy</li>
                    <li>Multi-camera tracking and re-identification</li>
                    <li>Virtual line crossing detection</li>
                    <li>Intelligent alert prioritization</li>
                    <li>Encrypted evidence storage</li>
                    <li>Comprehensive audit trails</li>
                    <li>Role-based access control</li>
                    <li>Performance monitoring</li>
                </ul>
            </div>
            
            <div class="card">
                <h3>🎮 Demo Controls</h3>
                <p>This is a live demonstration of Project Argus capabilities.</p>
                <div style="margin-top: 1rem;">
                    <button class="btn" onclick="simulateCrossing()">Simulate Border Crossing</button>
                    <button class="btn" onclick="simulateAlert()">Generate Alert</button>
                    <button class="btn" onclick="updateMetrics()">Update Metrics</button>
                </div>
                <div style="margin-top: 1rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 10px;">
                    <strong>Performance Metrics:</strong><br>
                    • Detection Latency: &lt; 300ms<br>
                    • False Positive Rate: &lt; 1%<br>
                    • System Uptime: 99.9%<br>
                    • Concurrent Cameras: 100+
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let alertCount = 2;
        let incidentCount = 1;
        let detectionCount = 47;
        
        function simulateCrossing() {
            detectionCount++;
            document.getElementById('detection-count').textContent = detectionCount;
            
            // Add visual feedback
            const cameras = document.querySelectorAll('.video-placeholder');
            cameras[0].style.borderColor = '#FF5722';
            setTimeout(() => {
                cameras[0].style.borderColor = '#4CAF50';
            }, 2000);
            
            // Generate alert after delay
            setTimeout(() => {
                simulateAlert();
            }, 1500);
        }
        
        function simulateAlert() {
            alertCount++;
            document.getElementById('alert-count').textContent = alertCount;
            
            const alertsContainer = document.querySelector('.card:nth-child(2)');
            const newAlert = document.createElement('div');
            newAlert.className = 'alert-item';
            newAlert.innerHTML = `
                <strong>SIMULATED CROSSING</strong><br>
                <small>Just now • Camera: Demo Camera</small><br>
                <span class="status alert">HIGH</span>
                Confidence: ${Math.floor(Math.random() * 20 + 80)}%
            `;
            
            const existingAlerts = alertsContainer.querySelector('.alert-item');
            if (existingAlerts) {
                alertsContainer.insertBefore(newAlert, existingAlerts);
            }
        }
        
        function simulateIncident() {
            incidentCount++;
            document.getElementById('incident-count').textContent = incidentCount;
            
            const incidentsContainer = document.querySelector('.card:nth-child(3)');
            const newIncident = document.createElement('div');
            newIncident.className = 'incident-item';
            newIncident.innerHTML = `
                <strong>Incident #INC-2024-${String(incidentCount).padStart(3, '0')}</strong><br>
                <small>New incident created from alert</small><br>
                <span class="status alert">OPEN</span>
                Priority: High
            `;
            
            const existingIncidents = incidentsContainer.querySelector('.incident-item');
            if (existingIncidents) {
                incidentsContainer.insertBefore(newIncident, existingIncidents);
            }
        }
        
        function updateMetrics() {
            detectionCount += Math.floor(Math.random() * 5 + 1);
            document.getElementById('detection-count').textContent = detectionCount;
            
            // Random chance for new alert
            if (Math.random() > 0.7) {
                simulateAlert();
            }
        }
        
        // Auto-update metrics
        setInterval(() => {
            if (Math.random() > 0.8) {
                updateMetrics();
            }
        }, 10000);
        
        // Initialize
        console.log('Project Argus Demo Loaded');
        console.log('System Status: All systems operational');
    </script>
</body>
</html>'''
    
    return html_content


def start_server():
    """Start the demo server."""
    class DemoHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/' or self.path == '/index.html':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                html_content = create_quick_demo_html()
                self.wfile.write(html_content.encode('utf-8'))
            else:
                super().do_GET()
    
    # Find available port
    for port in range(8082, 8090):
        try:
            with socketserver.TCPServer(("", port), DemoHandler) as httpd:
                print(f"🚀 Project Argus Demo Server running at http://localhost:{port}")
                print(f"🌐 Opening browser...")
                
                # Open browser
                threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{port}')).start()
                
                httpd.serve_forever()
                break
        except OSError:
            continue
    else:
        print("❌ Could not find available port")


if __name__ == "__main__":
    print("🎬 Starting Project Argus Quick Demo...")
    start_server()