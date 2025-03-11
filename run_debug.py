#!/usr/bin/env python3
"""
Debug run script with instrumentation for the social evolution simulator.
"""

import os
import sys
import time
import traceback
import signal
import atexit

# Register signal handlers for better debugging
def handle_segfault(signum, frame):
    print("\n==== SEGMENTATION FAULT DETECTED ====")
    print("Stack trace at the time of the fault:")
    traceback.print_stack(frame)
    print("=======================================\n")
    sys.exit(1)

# Register segfault handler
signal.signal(signal.SIGSEGV, handle_segfault)

# Apply fixes
print("Applying fixes before running...")
from emergency_fix import fix_simulation_class, add_metal_cleanup, fix_quadtree_issues
fix_simulation_class()
add_metal_cleanup()
fix_quadtree_issues()

# Set environment variables for better debugging
os.environ["PYTHONFAULTHANDLER"] = "1"  # Enable Python fault handler
os.environ["MPS_ENABLE_VALIDATION_LAYER"] = "1"  # Enable Metal validation if available

# Process command line arguments to modify for safety
if "--use-metal" in sys.argv and "--threads" not in sys.argv:
    # If using Metal but no thread count specified, set threads to 1 for safety
    print("WARNING: Using Metal acceleration with default threads. Setting --threads=1 for stability.")
    sys.argv.append("--threads=1")

# Import the main module and run with safety measures
print("Starting simulation with debug monitoring...")
try:
    from main import main
    sys.exit(main())
except Exception as e:
    print(f"\n==== UNHANDLED EXCEPTION ====")
    print(f"Error: {e}")
    print("Stack trace:")
    traceback.print_exc()
    print("=============================\n")
    sys.exit(1)
