#!/usr/bin/env python3
"""
Simple launcher to open Project Argus in browser
"""

import webbrowser
import time
import requests

def check_server():
    """Check if the server is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("🚀 Opening Project Argus Application...")
    
    if check_server():
        print("✅ Server is running")
        print("🌐 Opening browser...")
        
        # Try multiple ways to open browser
        urls = [
            "http://localhost:8000/dashboard",
            "http://localhost:8000",
            "http://127.0.0.1:8000/dashboard"
        ]
        
        for url in urls:
            try:
                webbrowser.open(url)
                print(f"📍 Opened: {url}")
                break
            except Exception as e:
                print(f"❌ Failed to open {url}: {e}")
                continue
        
        print("\n🎯 Available URLs:")
        print("   📊 Dashboard: http://localhost:8000/dashboard")
        print("   🔧 API: http://localhost:8000")
        print("   📖 Docs: http://localhost:8000/docs")
        print("\n💡 If browser didn't open automatically,")
        print("   copy and paste any URL above into your browser")
        
    else:
        print("❌ Server is not running!")
        print("💡 Please start the server first with:")
        print("   python simple_api_gateway.py")

if __name__ == "__main__":
    main()