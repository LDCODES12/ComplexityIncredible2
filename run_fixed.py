#!/usr/bin/env python3
"""
Run script for social evolution simulator with fixes for segmentation fault.
"""

import sys
import os
from pathlib import Path

# Add script directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import and apply fixes
try:
    from fix_all import fix_quadtree, fix_metal, fix_multiprocessing

    # Apply all fixes
    fixes_applied = []
    if fix_quadtree():
        fixes_applied.append("quadtree")
    if fix_metal():
        fixes_applied.append("metal")
    if fix_multiprocessing():
        fixes_applied.append("multiprocessing")

    if fixes_applied:
        print(f"Applied fixes: {', '.join(fixes_applied)}")
    else:
        print("No fixes were applied")
except ImportError as e:
    print(f"Could not import fixes: {e}")

# Run the main script with all arguments
try:
    from main import main
    sys.exit(main())
except ImportError:
    print("Could not import main module. Make sure you're in the correct directory.")
    sys.exit(1)
