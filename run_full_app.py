#!/usr/bin/env python3
"""
Full Project Argus Application Runner
Runs the complete border detection system with all services
"""

import os
import sys
import time
import subprocess
import threading
import signal
import webbrowser
from pathlib import Path
import requests

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class ProjectArgusRunner:
    """Main application runner for Project Argus."""
    
    def __init__(self):
        self.processes = []
        self.running = True
        self.setup_environment()
    
    def setup_environment(self):
        """Set up environment variables and configuration."""
        print("🔧 Setting up Project Argus environment...")
        
        # Set Python path
        os.environ['PYTHONPATH'] = str(project_root)
        
        # Create necessary directories
        directories = ['logs', 'data', 'models', 'uploads', 'evidence']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        
        # Set environment variables
        env_vars = {
            'ENVIRONMENT': 'production',
            'LOG_LEVEL': 'INFO',
            'DATABASE_URL': 'sqlite:///./data/project_argus.db',
            'REDIS_URL': 'redis://localhost:6379/0',
            'JWT_SECRET_KEY': 'your-secret-key-change-in-production',
            'ENCRYPTION_KEY': 'your-32-byte-encryption-key-here',
            'API_HOST': '0.0.0.0',
            'API_PORT': '8000',
            'CORS_ORIGINS': '["http://localhost:3000", "http://localhost:8000"]'
        }
        
        os.environ.update(env_vars)
        print("✅ Environment configured")
    
    def run_service(self, name, command, cwd=None, env=None):
        """Run a service as a background process."""
        print(f"🚀 Starting {name}...")
        
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd or project_root,
                env=full_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes.append((process, name))
            
            # Start output reader thread
            def read_output():
                for line in iter(process.stdout.readline, ''):
                    if line.strip():
                        print(f"[{name}] {line.strip()}")
                process.stdout.close()
            
            thread = threading.Thread(target=read_output, daemon=True)
            thread.start()
            
            return process
            
        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return None
    
    def start_api_gateway(self):
        """Start the main API Gateway."""
        return self.run_service(
            "API Gateway",
            f"python -m uvicorn services.api-gateway.main:app --host 0.0.0.0 --port 8000 --reload",
            env={'PYTHONPATH': str(project_root)}
        )
    
    def start_auth_service(self):
        """Start the Authentication Service."""
        return self.run_service(
            "Auth Service",
            "python main.py",
            cwd="services/auth-service",
            env={'PORT': '8001', 'PYTHONPATH': str(project_root)}
        )
    
    def start_alert_service(self):
        """Start the Alert Service."""
        return self.run_service(
            "Alert Service", 
            "python main.py",
            cwd="services/alert-service",
            env={'PORT': '8003', 'PYTHONPATH': str(project_root)}
        )
    
    def start_evidence_service(self):
        """Start the Evidence Service."""
        return self.run_service(
            "Evidence Service",
            "python main.py",
            cwd="services/evidence-service", 
            env={'PORT': '8005', 'PYTHONPATH': str(project_root)}
        )
    
    def start_tracking_service(self):
        """Start the Tracking Service."""
        return self.run_service(
            "Tracking Service",
            "python main.py",
            cwd="services/tracking-service",
            env={'PORT': '8004', 'PYTHONPATH': str(project_root)}
        )
    
    def start_edge_node(self):
        """Start the Edge Node."""
        return self.run_service(
            "Edge Node",
            "python main.py",
            cwd="edge",
            env={'PYTHONPATH': str(project_root)}
        )
    
    def start_dashboard(self):
        """Start the React Dashboard."""
        dashboard_path = project_root / "dashboard"
        
        # Check if node_modules exists
        if not (dashboard_path / "node_modules").exists():
            print("📦 Installing dashboard dependencies...")
            result = subprocess.run(['npm', 'install'], cwd=dashboard_path)
            if result.returncode != 0:
                print("❌ Failed to install dashboard dependencies")
                return None
        
        return self.run_service(
            "Dashboard",
            "npm start",
            cwd="dashboard",
            env={'BROWSER': 'none', 'PORT': '3000'}
        )
    
    def wait_for_service(self, url, service_name, timeout=30):
        """Wait for a service to be ready."""
        print(f"⏳ Waiting for {service_name} to be ready...")
        
        for i in range(timeout):
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    print(f"✅ {service_name} is ready")
                    return True
            except:
                pass
            time.sleep(1)
        
        print(f"⚠️  {service_name} not ready after {timeout}s")
        return False
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n🛑 Shutting down Project Argus...")
        self.running = False
        
        for process, name in self.processes:
            print(f"Stopping {name}...")
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        sys.exit(0)
    
    def run(self):
        """Run the complete Project Argus application."""
        print("🛡️  PROJECT ARGUS - BORDER DETECTION SYSTEM")
        print("=" * 60)
        print("🚀 Starting all services...")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Start services in order
        services_started = []
        
        # 1. Start API Gateway (core service)
        api_process = self.start_api_gateway()
        if api_process:
            services_started.append("API Gateway")
            time.sleep(3)
        
        # 2. Start Authentication Service
        auth_process = self.start_auth_service()
        if auth_process:
            services_started.append("Auth Service")
            time.sleep(2)
        
        # 3. Start Alert Service
        alert_process = self.start_alert_service()
        if alert_process:
            services_started.append("Alert Service")
            time.sleep(2)
        
        # 4. Start Evidence Service
        evidence_process = self.start_evidence_service()
        if evidence_process:
            services_started.append("Evidence Service")
            time.sleep(2)
        
        # 5. Start Tracking Service
        tracking_process = self.start_tracking_service()
        if tracking_process:
            services_started.append("Tracking Service")
            time.sleep(2)
        
        # 6. Start Edge Node
        edge_process = self.start_edge_node()
        if edge_process:
            services_started.append("Edge Node")
            time.sleep(2)
        
        # 7. Start Dashboard
        dashboard_process = self.start_dashboard()
        if dashboard_process:
            services_started.append("Dashboard")
            time.sleep(5)
        
        # Wait for core services to be ready
        self.wait_for_service("http://localhost:8000/health", "API Gateway")
        
        print("\n" + "=" * 60)
        print("🎉 PROJECT ARGUS IS NOW RUNNING!")
        print("=" * 60)
        
        print(f"\n📊 Services Started: {len(services_started)}")
        for service in services_started:
            print(f"   ✅ {service}")
        
        print(f"\n🌐 Access Points:")
        print(f"   📊 Main Dashboard:     http://localhost:3000")
        print(f"   🔧 API Gateway:        http://localhost:8000")
        print(f"   📖 API Documentation:  http://localhost:8000/docs")
        print(f"   🔐 Auth Service:       http://localhost:8001")
        print(f"   🚨 Alert Service:      http://localhost:8003")
        print(f"   📁 Evidence Service:   http://localhost:8005")
        print(f"   🎯 Tracking Service:   http://localhost:8004")
        
        print(f"\n🎯 Key Features:")
        print(f"   • AI-powered person detection")
        print(f"   • Multi-camera tracking")
        print(f"   • Real-time alerts")
        print(f"   • Evidence management")
        print(f"   • Incident workflows")
        print(f"   • Security compliance")
        
        print(f"\n🛑 Press Ctrl+C to stop all services")
        
        # Try to open browser
        try:
            time.sleep(2)
            webbrowser.open('http://localhost:3000')
            print(f"🌐 Opening dashboard in browser...")
        except:
            pass
        
        # Keep running
        try:
            while self.running:
                time.sleep(1)
                
                # Check if critical processes are still running
                for process, name in self.processes:
                    if process.poll() is not None:
                        print(f"⚠️  {name} has stopped")
                        
        except KeyboardInterrupt:
            self.signal_handler(signal.SIGINT, None)
        
        return 0


def main():
    """Main entry point."""
    runner = ProjectArgusRunner()
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())