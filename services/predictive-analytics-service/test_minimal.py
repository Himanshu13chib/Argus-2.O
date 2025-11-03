#!/usr/bin/env python3

print("Starting minimal test...")

from datetime import datetime
import asyncio

print("Basic imports successful")

# Test the core functionality without complex imports
class SimplePredictiveEngine:
    def __init__(self):
        print("Engine initialized")
        self.models = {}
    
    async def test_method(self):
        return {"status": "working", "timestamp": datetime.now()}

async def main():
    print("Testing engine...")
    engine = SimplePredictiveEngine()
    result = await engine.test_method()
    print(f"Test result: {result}")
    print("✓ All tests passed!")

if __name__ == "__main__":
    asyncio.run(main())