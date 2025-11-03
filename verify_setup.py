#!/usr/bin/env python3
"""
Project Argus Setup Verification Script
Verifies that all core components are properly configured.
"""

import sys
import os
from pathlib import Path

def check_directory_structure():
    """Check if all required directories exist."""
    required_dirs = [
        'shared/models',
        'shared/interfaces', 
        'edge',
        'dashboard',
        'services/api-gateway',
        'services/alert-service',
        'services/tracking-service',
        'services/evidence-service',
        'infrastructure/database',
        'logs',
        'data',
        'models'
    ]
    
    print("Checking directory structure...")
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"  ✓ {dir_path}")
    
    if missing_dirs:
        print(f"  ✗ Missing directories: {missing_dirs}")
        return False
    
    print("  ✓ All required directories exist")
    return True

def check_core_models():
    """Check if core models can be imported."""
    print("\nChecking core models...")
    
    try:
        sys.path.append('.')
        from shared.models import (
            Detection, Track, Alert, Incident, Evidence, 
            Camera, VirtualLine, HealthStatus, User
        )
        print("  ✓ All core models imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import models: {e}")
        return False

def check_core_interfaces():
    """Check if core interfaces can be imported."""
    print("\nChecking core interfaces...")
    
    try:
        from shared.interfaces.detection import IDetectionPipeline
        from shared.interfaces.tracking import IMultiCameraTracker
        from shared.interfaces.alerts import IAlertEngine
        from shared.interfaces.incidents import IIncidentManager
        from shared.interfaces.evidence import IEvidenceStore
        from shared.interfaces.health import IHealthMonitor
        from shared.interfaces.security import ISecurityManager
        print("  ✓ All core interfaces imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import interfaces: {e}")
        return False

def check_configuration_files():
    """Check if configuration files exist."""
    print("\nChecking configuration files...")
    
    config_files = [
        '.env',
        'docker-compose.yml',
        'requirements.txt',
        'dashboard/package.json',
        'Makefile'
    ]
    
    missing_files = []
    for file_path in config_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"  ✓ {file_path}")
    
    if missing_files:
        print(f"  ✗ Missing files: {missing_files}")
        return False
    
    print("  ✓ All configuration files exist")
    return True

def check_service_structure():
    """Check if service files are properly structured."""
    print("\nChecking service structure...")
    
    services = ['api-gateway', 'alert-service', 'tracking-service', 'evidence-service']
    
    for service in services:
        service_dir = Path(f'services/{service}')
        main_file = service_dir / 'main.py'
        dockerfile = service_dir / 'Dockerfile'
        requirements = service_dir / 'requirements.txt'
        
        if not main_file.exists():
            print(f"  ✗ Missing {main_file}")
            return False
        if not dockerfile.exists():
            print(f"  ✗ Missing {dockerfile}")
            return False
        if not requirements.exists():
            print(f"  ✗ Missing {requirements}")
            return False
        
        print(f"  ✓ {service} service structure complete")
    
    return True

def main():
    """Main verification function."""
    print("Project Argus Setup Verification")
    print("=" * 40)
    
    checks = [
        check_directory_structure,
        check_core_models,
        check_core_interfaces,
        check_configuration_files,
        check_service_structure
    ]
    
    all_passed = True
    
    for check in checks:
        if not check():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All verification checks passed!")
        print("✓ Project Argus is properly set up and ready for development")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up dashboard: cd dashboard && npm install")
        print("3. Start development: make dev-up (requires Docker)")
        return 0
    else:
        print("✗ Some verification checks failed")
        print("Please review the errors above and fix any issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())