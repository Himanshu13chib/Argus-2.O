#!/usr/bin/env python3
"""
Simple Edge Node Simulator for Project Argus
Simulates camera feeds and detection events
"""

import asyncio
import json
import time
import random
from datetime import datetime
import requests
from typing import Dict, Any


class EdgeSimulator:
    """Simulates edge node behavior."""
    
    def __init__(self, api_gateway_url: str = "http://localhost:8000"):
        self.api_gateway_url = api_gateway_url
        self.camera_id = "edge_camera_001"
        self.running = False
        
    async def simulate_detection_events(self):
        """Simulate detection events."""
        print(f"🎥 Starting edge simulation for {self.camera_id}")
        
        detection_count = 0
        
        while self.running:
            try:
                # Simulate random detection events
                if random.random() > 0.7:  # 30% chance of detection
                    detection_count += 1
                    
                    detection = {
                        "id": f"det_{self.camera_id}_{detection_count:04d}",
                        "camera_id": self.camera_id,
                        "timestamp": datetime.now().isoformat(),
                        "confidence": round(random.uniform(0.7, 0.95), 2),
                        "bbox": {
                            "x": random.randint(50, 300),
                            "y": random.randint(50, 200),
                            "width": random.randint(30, 80),
                            "height": random.randint(60, 120)
                        },
                        "detection_class": "person",
                        "metadata": {
                            "lighting": random.choice(["daylight", "dawn", "night"]),
                            "weather": random.choice(["clear", "cloudy", "foggy"])
                        }
                    }
                    
                    print(f"📹 Detection {detection_count}: Person detected (confidence: {detection['confidence']})")
                    
                    # In a real system, this would send to the API gateway
                    # For now, just log the detection
                    
                await asyncio.sleep(random.uniform(2, 8))  # Random interval between detections
                
            except Exception as e:
                print(f"❌ Error in detection simulation: {e}")
                await asyncio.sleep(5)
    
    async def simulate_health_monitoring(self):
        """Simulate health monitoring."""
        while self.running:
            try:
                health_status = {
                    "camera_id": self.camera_id,
                    "timestamp": datetime.now().isoformat(),
                    "status": "healthy",
                    "metrics": {
                        "cpu_usage": round(random.uniform(10, 40), 1),
                        "memory_usage": round(random.uniform(20, 60), 1),
                        "temperature": round(random.uniform(35, 55), 1),
                        "fps": random.randint(25, 30),
                        "latency_ms": random.randint(80, 200)
                    }
                }
                
                print(f"💓 Health: CPU {health_status['metrics']['cpu_usage']}%, "
                      f"Mem {health_status['metrics']['memory_usage']}%, "
                      f"Temp {health_status['metrics']['temperature']}°C")
                
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except Exception as e:
                print(f"❌ Error in health monitoring: {e}")
                await asyncio.sleep(10)
    
    async def check_api_gateway(self):
        """Check if API gateway is available."""
        try:
            response = requests.get(f"{self.api_gateway_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ API Gateway is available at {self.api_gateway_url}")
                return True
            else:
                print(f"⚠️  API Gateway returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot reach API Gateway: {e}")
            return False
    
    async def start(self):
        """Start the edge simulator."""
        print("🚀 Starting Project Argus Edge Simulator...")
        
        # Check API gateway connectivity
        if await asyncio.to_thread(self.check_api_gateway):
            print("🔗 Connected to API Gateway")
        else:
            print("⚠️  Running in offline mode")
        
        self.running = True
        
        # Start simulation tasks
        tasks = [
            asyncio.create_task(self.simulate_detection_events()),
            asyncio.create_task(self.simulate_health_monitoring())
        ]
        
        print(f"📡 Edge node simulation active")
        print(f"🎯 Camera: {self.camera_id}")
        print(f"🔄 Generating detection events...")
        print(f"🛑 Press Ctrl+C to stop")
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\n🛑 Stopping edge simulator...")
            self.running = False
            
            # Cancel tasks
            for task in tasks:
                task.cancel()
            
            print("✅ Edge simulator stopped")


async def main():
    """Main entry point."""
    simulator = EdgeSimulator()
    await simulator.start()


if __name__ == "__main__":
    asyncio.run(main())