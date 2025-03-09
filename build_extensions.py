#!/usr/bin/env python3
"""
Build script for compiling Cython extensions in the social evolution simulator.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def is_apple_silicon():
    """Check if we're running on Apple Silicon."""
    return (
        platform.system() == "Darwin" and
        platform.processor() == "arm" and
        platform.machine() == "arm64"
    )

def main():
    """Main build function."""
    print("Building Cython extensions for the social evolution simulator...")

    # Set compilation options based on platform
    compile_args = ["-O3"]
    link_args = []

    # Add platform-specific options
    if platform.system() == "Darwin":  # macOS
        # Detect CPU architecture
        if is_apple_silicon():
            print("Detected Apple Silicon - optimizing for M1/M2/M3...")
            compile_args.extend(["-arch", "arm64"])
            link_args.extend(["-arch", "arm64"])
        else:
            print("Detected Intel Mac...")
            compile_args.extend(["-arch", "x86_64"])
            link_args.extend(["-arch", "x86_64"])

    # Check if OpenMP is available
    has_openmp = False
    try:
        if platform.system() == "Darwin":
            # macOS requires special handling for OpenMP
            if subprocess.call(["brew", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0:
                print("Homebrew detected, checking for libomp...")
                if subprocess.call(["brew", "list", "libomp"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0:
                    print("OpenMP support detected via Homebrew libomp")
                    has_openmp = True
                    # Add paths for libomp from Homebrew
                    compile_args.extend(["-Xpreprocessor", "-fopenmp", "-I/usr/local/opt/libomp/include"])
                    link_args.extend(["-Xpreprocessor", "-fopenmp", "-L/usr/local/opt/libomp/lib", "-lomp"])
        else:
            # Linux or Windows
            has_openmp = True
            compile_args.append("-fopenmp")
            link_args.append("-fopenmp")
    except:
        print("Could not detect OpenMP support")

    if not has_openmp:
        print("Building without OpenMP support. Parallel processing will be limited.")

    # Prepare build command
    build_cmd = [
        sys.executable, 
        "setup.py", 
        "build_ext", 
        "--inplace"
    ]

    # Set environment variables for compilation flags
    env = os.environ.copy()
    env["CFLAGS"] = " ".join(compile_args)
    env["LDFLAGS"] = " ".join(link_args)

    # Run build
    print("Running build command:", " ".join(build_cmd))
    try:
        subprocess.run(build_cmd, env=env, check=True)
        print("Build completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        return 1

    print("\nTesting compiled extensions...")
    # Test each extension
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
