#!/usr/bin/env python3
"""
Development Environment Setup Script for Project Argus
"""

import os
import sys
import subprocess
import venv
from pathlib import Path


def run_command(command, cwd=None, check=True):
    """Run a shell command."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, check=check)
    return result.returncode == 0


def setup_python_venv():
    """Set up Python virtual environment."""
    print("Setting up Python virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("Virtual environment already exists")
        return True
    
    # Create virtual environment
    venv.create("venv", with_pip=True)
    
    # Determine activation script path
    if sys.platform == "win32":
        activate_script = "venv\\Scripts\\activate.bat"
        pip_path = "venv\\Scripts\\pip"
    else:
        activate_script = "venv/bin/activate"
        pip_path = "venv/bin/pip"
    
    print(f"Virtual environment created. Activate with: {activate_script}")
    
    # Install base requirements
    if not run_command(f"{pip_path} install --upgrade pip"):
        return False
    
    if not run_command(f"{pip_path} install -r requirements.txt"):
        return False
    
    return True


def setup_node_env():
    """Set up Node.js environment for dashboard."""
    print("Setting up Node.js environment...")
    
    dashboard_path = Path("dashboard")
    if not dashboard_path.exists():
        print("Dashboard directory not found")
        return False
    
    # Check if node_modules exists
    node_modules = dashboard_path / "node_modules"
    if node_modules.exists():
        print("Node modules already installed")
        return True
    
    # Install npm dependencies
    if not run_command("npm install", cwd=dashboard_path):
        return False
    
    return True


def setup_docker_env():
    """Set up Docker environment."""
    print("Setting up Docker environment...")
    
    # Check if Docker is available
    if not run_command("docker --version", check=False):
        print("Docker not found. Please install Docker first.")
        return False
    
    # Check if docker-compose is available
    if not run_command("docker-compose --version", check=False):
        print("docker-compose not found. Please install docker-compose first.")
        return False
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        print("Creating .env file from template...")
        env_example = Path(".env.example")
        if env_example.exists():
            env_file.write_text(env_example.read_text())
        else:
            # Create basic .env file
            env_content = """# Project Argus Environment Variables
POSTGRES_PASSWORD=argus_secure_pass
REDIS_PASSWORD=redis_secure_pass
MINIO_ROOT_USER=argus_admin
MINIO_ROOT_PASSWORD=minio_secure_pass

# Development settings
ENVIRONMENT=development
LOG_LEVEL=DEBUG
"""
            env_file.write_text(env_content)
    
    return True


def create_directories():
    """Create necessary directories."""
    print("Creating project directories...")
    
    directories = [
        "logs",
        "data",
        "models",
        "edge/models",
        "edge/data",
        "edge/logs",
        "dashboard/public/assets",
        "infrastructure/monitoring",
        "infrastructure/scripts",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    return True


def main():
    """Main setup function."""
    print("Setting up Project Argus development environment...")
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("Python 3.9 or higher is required")
        return 1
    
    # Create directories
    if not create_directories():
        print("Failed to create directories")
        return 1
    
    # Set up Python environment
    if not setup_python_venv():
        print("Failed to set up Python environment")
        return 1
    
    # Set up Node.js environment
    if not setup_node_env():
        print("Failed to set up Node.js environment")
        return 1
    
    # Set up Docker environment
    if not setup_docker_env():
        print("Failed to set up Docker environment")
        return 1
    
    print("\n" + "="*60)
    print("Development environment setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Activate Python virtual environment:")
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate.bat")
    else:
        print("   source venv/bin/activate")
    print("\n2. Start development services:")
    print("   docker-compose up -d")
    print("\n3. Start dashboard development server:")
    print("   cd dashboard && npm start")
    print("\n4. Start edge node (in development mode):")
    print("   python edge/main.py --debug")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())