#!/usr/bin/env python3
"""
Test runner for alert service with proper path setup.
"""

import sys
import os
from pathlib import Path

# Add workspace root to Python path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

# Now run the tests
if __name__ == "__main__":
    import pytest
    
    # Run tests with proper configuration
    exit_code = pytest.main([
        "tests/",
        "-v",
        "--tb=short",
        "--strict-markers"
    ])
    
    sys.exit(exit_code)