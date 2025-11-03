#!/usr/bin/env python3
"""
Project Argus Application Launcher
Opens the real Project Argus application in your browser
"""

import webbrowser
import time
import requests
import sys

def check_services():
    """Check which services are running."""
    services = {
        "API Gateway": "http://localhost:8000/health",
        "Dashboard": "http://localhost:3000",
        "Alert Service": "http://localhost:8003/health",
        "Auth Service": "http://localhost:8001/health",
        "Evidence Service": "http://localhost:8005/health",
        "Tracking Service": "http://localhost:8004/health"
    }
    
    running_services = []
    
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                running_services.append((name, url))
                print(f"✅ {name}: Running")
            else:
                print(f"⚠️  {name}: Not responding")
        except:
            print(f"❌ {name}: Not available")
    
    return running_services

def main():
    """Launch the Project Argus application."""
    print("🛡️  PROJECT ARGUS - APPLICATION LAUNCHER")
    print("=" * 50)
    
    print("🔍 Checking service status...")
    running_services = check_services()
    
    if not running_services:
        print("\n❌ No services are running!")
        print("\n💡 To start Project Argus, run:")
        print("   python run_full_app.py")
        return 1
    
    print(f"\n✅ Found {len(running_services)} running services")
    
    # Determine best URL to open
    if any("Dashboard" in service[0] for service in running_services):
        main_url = "http://localhost:3000"
        app_name = "React Dashboard"
    elif any("API Gateway" in service[0] for service in running_services):
        main_url = "http://localhost:8000"
        app_name = "API Gateway"
    else:
        main_url = running_services[0][1]
        app_name = running_services[0][0]
    
    print(f"\n🌐 Opening {app_name}...")
    print(f"📍 URL: {main_url}")
    
    # Open browser
    try:
        webbrowser.open(main_url)
        print("✅ Browser opened successfully")
    except Exception as e:
        print(f"❌ Could not open browser: {e}")
    
    print(f"\n🎯 Available Services:")
    for name, url in running_services:
        print(f"   • {name}: {url}")
    
    print(f"\n💡 If browser didn't open, manually visit:")
    print(f"   {main_url}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())