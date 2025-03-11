#!/usr/bin/env python3
"""
Rebuild Cython extensions with debugging and bounds checking enabled.
"""

import os
import sys
import subprocess
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

print("Rebuilding Cython extensions with debugging enabled...")

# Define extensions with debugging directives
extensions = [
    Extension(
        "social.interactions",
        ["social/interactions.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "simulation.spatial.quadtree",
        ["simulation/spatial/quadtree.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

# Apply debug directives
debug_directives = {
    "boundscheck": True,       # Check array bounds
    "initializedcheck": True,  # Check if memoryviews are initialized
    "wraparound": True,        # Check negative indexing
    "cdivision": False,        # Check division by zero
    "language_level": 3,       # Python 3
    "profile": True,           # Enable profiling
    "linetrace": True,         # Enable line tracing for coverage
}

# Build extensions
setup(
    name="social_evolution_simulator_debug",
    version="2.0.0",
    description="Debug build of social evolution simulator",
    ext_modules=cythonize(extensions, compiler_directives=debug_directives),
)

print("\nCython extensions rebuilt with debugging enabled.")
print("Run the simulation with: python run_debug.py")

# Optionally add a test to verify the rebuilt modules
try:
    print("\nTesting rebuilt modules...")

    # Test social.interactions
    test_cmd = [sys.executable, "-c", "from social.interactions import calculate_relationship_strength; print('social.interactions loaded successfully')"]
    subprocess.run(test_cmd, check=True)

    # Test simulation.spatial.quadtree
    test_cmd = [sys.executable, "-c", "from simulation.spatial.quadtree import create_quadtree; print('simulation.spatial.quadtree loaded successfully')"]
    subprocess.run(test_cmd, check=True)

    print("All modules loaded successfully!")
except subprocess.CalledProcessError as e:
    print(f"Error testing rebuilt modules: {e}")
except ImportError as e:
    print(f"Import error: {e}")
