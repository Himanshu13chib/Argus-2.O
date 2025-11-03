#!/usr/bin/env python3
"""
Development Runner for Project Argus
Runs the application in development mode without Docker
"""

import os
import sys
import time
import subprocess
import threading
import signal
from pathlib import Path


class DevRunner:
    """Development environment runner."""
    
    def __init__(self):
        self.processes = []
        self.running = True
        
    def run_command_async(self, command, name, cwd=None, env=None):
        """Run a command asynchronously."""
        print(f"🚀 Starting {name}...")
        
        # Merge environment variables
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                env=full_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes.append((process, name))
            
            # Start thread to read output
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
    
    def check_dependencies(self):
        """Check if required dependencies are available."""
        print("🔍 Checking dependencies...")
        
        # Check Python
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ is required")
            return False
        print("✅ Python version OK")
        
        # Check if we can import required packages
        try:
            import fastapi
            import uvicorn
            print("✅ FastAPI available")
        except ImportError:
            print("❌ FastAPI not installed. Run: pip install fastapi uvicorn")
            return False
        
        # Check Node.js for dashboard
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Node.js available: {result.stdout.strip()}")
            else:
                print("⚠️  Node.js not found - dashboard won't be available")
        except FileNotFoundError:
            print("⚠️  Node.js not found - dashboard won't be available")
        
        return True
    
    def setup_environment(self):
        """Set up development environment."""
        print("🔧 Setting up environment...")
        
        # Create necessary directories
        directories = ['logs', 'data', 'models']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        
        # Set environment variables for development
        os.environ.update({
            'ENVIRONMENT': 'development',
            'LOG_LEVEL': 'INFO',
            'DATABASE_URL': 'sqlite:///./data/argus_dev.db',
            'REDIS_URL': 'redis://localhost:6379/0',
            'API_HOST': '0.0.0.0',
            'API_PORT': '8000',
        })
        
        print("✅ Environment setup complete")
    
    def start_api_gateway(self):
        """Start the API Gateway service."""
        return self.run_command_async(
            "python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload",
            "API Gateway",
            cwd="services/api-gateway"
        )
    
    def start_alert_service(self):
        """Start the Alert service."""
        return self.run_command_async(
            "python main.py",
            "Alert Service", 
            cwd="services/alert-service",
            env={'PORT': '8003'}
        )
    
    def start_dashboard(self):
        """Start the React dashboard."""
        dashboard_path = Path("dashboard")
        
        # Check if node_modules exists
        if not (dashboard_path / "node_modules").exists():
            print("📦 Installing dashboard dependencies...")
            result = subprocess.run(['npm', 'install'], cwd=dashboard_path)
            if result.returncode != 0:
                print("❌ Failed to install dashboard dependencies")
                return None
        
        return self.run_command_async(
            "npm start",
            "Dashboard",
            cwd="dashboard",
            env={'BROWSER': 'none'}  # Don't auto-open browser
        )
    
    def start_edge_simulator(self):
        """Start the edge node simulator."""
        return self.run_command_async(
            "python main.py --simulation",
            "Edge Simulator",
            cwd="edge"
        )
    
    def start_demo_environment(self):
        """Start demo environment with sample data."""
        return self.run_command_async(
            "python tests/integration/test_demo_environment.py",
            "Demo Environment",
            cwd="."
        )
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n🛑 Shutting down services...")
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
        """Run the development environment."""
        print("🎯 Project Argus - Development Mode")
        print("=" * 50)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Check dependencies
        if not self.check_dependencies():
            return 1
        
        # Setup environment
        self.setup_environment()
        
        # Start services
        print("\n🚀 Starting services...")
        
        # Start API Gateway
        api_process = self.start_api_gateway()
        if api_process:
            time.sleep(2)  # Give it time to start
        
        # Start Alert Service
        alert_process = self.start_alert_service()
        if alert_process:
            time.sleep(1)
        
        # Start Dashboard
        dashboard_process = self.start_dashboard()
        if dashboard_process:
            time.sleep(3)  # Dashboard takes longer to start
        
        # Start Edge Simulator
        edge_process = self.start_edge_simulator()
        
        print("\n" + "=" * 50)
        print("🎉 Project Argus Development Environment Started!")
        print("=" * 50)
        print("\n📍 Available Services:")
        print("   🌐 API Gateway:    http://localhost:8000")
        print("   📊 Dashboard:      http://localhost:3000")
        print("   🚨 Alert Service:  http://localhost:8003")
        print("   📹 Edge Simulator: Running in background")
        print("\n📖 API Documentation: http://localhost:8000/docs")
        print("🔍 Health Check:      http://localhost:8000/health")
        print("\n⚠️  Note: This is a development environment.")
        print("   Some features may require additional setup.")
        print("\n🛑 Press Ctrl+C to stop all services")
        
        # Keep running until interrupted
        try:
            while self.running:
                time.sleep(1)
                
                # Check if any critical processes have died
                for process, name in self.processes:
                    if process.poll() is not None:
                        print(f"⚠️  {name} has stopped")
                        
        except KeyboardInterrupt:
            self.signal_handler(signal.SIGINT, None)
        
        return 0


def main():
    """Main entry point."""
    runner = DevRunner()
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())