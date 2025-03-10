#!/usr/bin/env python3
"""
Build script for compiling Cython extensions with proper OpenMP support on Apple Silicon.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def get_homebrew_prefix():
    """Get the Homebrew prefix directory based on architecture."""
    try:
        result = subprocess.run(["brew", "--prefix"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except:
        # Fallback to architecture-specific defaults
        if platform.machine() == "arm64":  # Apple Silicon
            return "/opt/homebrew"
        else:  # Intel Mac
            return "/usr/local"


def main():
    """Main build function with proper OpenMP paths."""
    print("Building Cython extensions for the social evolution simulator...")

    # Determine if we're on Apple Silicon
    is_apple_silicon = platform.machine() == "arm64"
    if is_apple_silicon:
        print("Detected Apple Silicon Mac")
    else:
        print("Detected Intel Mac")

    # Get the correct Homebrew prefix
    brew_prefix = get_homebrew_prefix()
    print(f"Using Homebrew prefix: {brew_prefix}")

    # Set up OpenMP paths
    omp_include = f"{brew_prefix}/opt/libomp/include"
    omp_lib = f"{brew_prefix}/opt/libomp/lib"

    # Verify libomp exists
    if os.path.exists(omp_include) and os.path.exists(f"{omp_lib}/libomp.dylib"):
        print(f"Found OpenMP at: {omp_include}")
        has_openmp = True
    else:
        print("Could not find OpenMP installation. Building without parallel processing.")
        has_openmp = False

    # Create a custom setup.py file
    with open("temp_setup.py", "w") as f:
        f.write("""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define extensions
extensions = [
    Extension(
        "social.interactions",
        ["social/interactions.pyx"],
        include_dirs=[np.get_include()],
        # Disable NumPy API warnings
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "simulation.spatial.quadtree",
        ["simulation/spatial/quadtree.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    name="social_evolution_simulator",
    version="2.0.0",
    description="Advanced social evolution simulator",
    ext_modules=cythonize(extensions, language_level=3),
)
        """)

    # Set environment variables for compilation flags
    env = os.environ.copy()

    # Basic optimization flags
    compile_flags = ["-O3"]
    link_flags = []

    # Architecture-specific flags
    if is_apple_silicon:
        compile_flags.append("-arch arm64")
        link_flags.append("-arch arm64")
    else:
        compile_flags.append("-arch x86_64")
        link_flags.append("-arch x86_64")

    # Add OpenMP flags if available
    if has_openmp:
        compile_flags.extend(["-Xpreprocessor", "-fopenmp", f"-I{omp_include}"])
        link_flags.extend(["-Xpreprocessor", "-fopenmp", f"-L{omp_lib}", "-lomp"])
    else:
        # Define macro to disable OpenMP in Cython
        compile_flags.append("-DCYTHON_WITHOUT_OPENMP")

    # Set environment variables
    env["CFLAGS"] = " ".join(compile_flags)
    env["LDFLAGS"] = " ".join(link_flags)

    # Print the commands we're using
    print(f"CFLAGS: {env['CFLAGS']}")
    print(f"LDFLAGS: {env['LDFLAGS']}")

    # Build command
    build_cmd = [
        sys.executable,
        "temp_setup.py",
        "build_ext",
        "--inplace"
    ]

    # Run build
    print("Running build command:", " ".join(build_cmd))
    try:
        subprocess.run(build_cmd, env=env, check=True)
        print("Build completed successfully!")
        # Clean up the temporary setup file
        os.remove("temp_setup.py")
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        return 1

    # Test the extensions
    print("\nTesting compiled extensions...")
    tests = [
        "from social.interactions import calculate_relationship_strength",
        "from simulation.spatial.quadtree import create_quadtree"
    ]

    for test in tests:
        print(f"Testing: {test}")
        try:
            subprocess.run([sys.executable, "-c", test], check=True)
            print("  ✓ Success")
        except subprocess.CalledProcessError:
            print("  ✗ Failed")

    print("\nSetup complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())