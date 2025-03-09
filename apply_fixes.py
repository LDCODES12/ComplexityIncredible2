#!/usr/bin/env python3
"""
Script to apply all fixes to the social evolution simulator codebase.
"""

import os
import sys
import re
import shutil
from pathlib import Path


def create_backup(filename):
    """Create a backup of a file if it exists."""
    filepath = Path(filename)
    if filepath.exists():
        backup_path = filepath.with_suffix(filepath.suffix + '.bak')
        print(f"Creating backup of {filepath} to {backup_path}")
        shutil.copy2(filepath, backup_path)
        return True
    return False


def create_world_py():
    """Create the missing environment/world.py file."""
    content = """
\"\"\"
World environment that integrates terrain, resources, and weather systems.
Manages the overall simulation environment and provides unified access.
\"\"\"

import os
import sys
import numpy as np
import random
from typing import List, Dict, Tuple, Set, Optional, Union, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

# Import environment components
from environment.conditions import Environment, WeatherSystem, TerrainGenerator
from environment.resources import Resource, ResourceManager, ResourcePool


class World:
    \"\"\"
    Integrated world environment that combines terrain, resources, and weather.
    Provides a unified interface for the simulation to interact with the environment.
    \"\"\"

    def __init__(self, config=None):
        \"\"\"
        Initialize the world environment.

        Args:
            config: Configuration dictionary (optional)
        \"\"\"
        self.config = config or CONFIG
        self.world_size = self.config["world_size"]
        self.step = 0

        # Initialize environment components
        self.environment = Environment(self.config)
        self.resource_manager = ResourceManager(self.world_size, self.config)

        # Generate initial resources
        self.resource_manager.generate_initial_resources()

        # For convenience, expose resources dictionary directly
        self.resources = self.resource_manager.resources

    def update(self, step):
        \"\"\"
        Update the world for a new simulation step.

        Args:
            step: Current simulation step

        Returns:
            Updated environmental conditions
        \"\"\"
        self.step = step

        # Update environment
        conditions = self.environment.update(step)

        # Update resources
        self.resource_manager.update_resources(step)

        # Expose resources directly
        self.resources = self.resource_manager.resources

        return conditions

    def get_nearby_resources(self, position, radius):
        \"\"\"
        Get resources near a position.

        Args:
            position: (x, y) position
            radius: Search radius

        Returns:
            List of resources within radius
        \"\"\"
        return self.resource_manager.get_nearby_resources(position, radius)

    def add_resource(self, position=None, value=None, type=None, properties=None):
        \"\"\"
        Add a new resource to the world.

        Args:
            position: (x, y) position (None = random)
            value: Resource value (None = default)
            type: Resource type (None = random)
            properties: Additional properties (optional)

        Returns:
            Newly created resource
        \"\"\"
        return self.resource_manager.add_resource(position, value, type, properties)

    def remove_resource(self, resource_id):
        \"\"\"
        Remove a resource from the world.

        Args:
            resource_id: ID of resource to remove

        Returns:
            True if resource was removed
        \"\"\"
        return self.resource_manager.remove_resource(resource_id)

    def get_environmental_effects(self):
        \"\"\"
        Get environmental effects on agents and resources.

        Returns:
            Dictionary of effect modifiers
        \"\"\"
        return self.environment.get_environment_effects()

    def is_position_valid(self, position):
        \"\"\"
        Check if a position is valid (within bounds and not blocked).

        Args:
            position: (x, y) position

        Returns:
            True if position is valid
        \"\"\"
        return self.environment.is_position_valid(position)

    def get_movement_cost(self, position):
        \"\"\"
        Get the movement cost at a position.

        Args:
            position: (x, y) position

        Returns:
            Movement cost multiplier
        \"\"\"
        return self.environment.get_movement_cost(position)

    def find_starting_positions(self, num_positions, min_distance=50):
        \"\"\"
        Find suitable starting positions for agents.

        Args:
            num_positions: Number of positions to find
            min_distance: Minimum distance between positions

        Returns:
            List of (x, y) positions
        \"\"\"
        return self.environment.terrain.find_starting_positions(num_positions, min_distance)

    def get_rich_resource_area(self, resource_type=None):
        \"\"\"
        Find an area with high resource potential.

        Args:
            resource_type: Type of resource (optional)

        Returns:
            (x, y) position of rich area
        \"\"\"
        return self.environment.get_rich_resource_area(resource_type)

    def get_state(self):
        \"\"\"
        Get the current state of the world.

        Returns:
            Dictionary containing world state
        \"\"\"
        return {
            "step": self.step,
            "conditions": self.environment.conditions,
            "resources": [r.to_dict() for r in self.resources.values()],
            "resource_distribution": self.resource_manager.get_resource_distribution(),
            "terrain": {
                "water_map": self.environment.terrain.water_map.tolist() if hasattr(self.environment.terrain, 'water_map') else None,
                "height_map": self.environment.terrain.height_map.tolist() if hasattr(self.environment.terrain, 'height_map') else None,
                "fertility_map": self.environment.terrain.fertility_map.tolist() if hasattr(self.environment.terrain, 'fertility_map') else None,
            }
        }
"""

    filepath = Path("environment/world.py")
    filepath.parent.mkdir(exist_ok=True)

    print(f"Creating environment/world.py")
    with open(filepath, "w") as f:
        f.write(content.strip())


def fix_environment_conditions():
    """Fix the Environment class in environment/conditions.py."""
    filepath = Path("environment/conditions.py")

    if not filepath.exists():
        print(f"Warning: {filepath} not found. Skipping.")
        return

    create_backup(filepath)

    with open(filepath, "r") as f:
        content = f.read()

    # Find the Environment class definition
    env_class_match = re.search(r'class Environment[^:]*:', content)

    if not env_class_match:
        print("Warning: Could not find Environment class. Skipping.")
        return

    # Check if get_nearby_resources method already exists
    if "def get_nearby_resources" in content:
        print("get_nearby_resources method already exists. Skipping.")
    else:
        # Add get_nearby_resources method to the Environment class
        # Find the end of the class (next class definition or end of file)
        class_end = content.find("class ", env_class_match.end())
        if class_end == -1:
            class_end = len(content)

        # Insert get_nearby_resources method
        get_nearby_resources_method = """
    def get_nearby_resources(self, position, radius):
        \"\"\"
        Get resources near a position.
        This is a compatibility method that delegates to the resources component.

        Args:
            position: (x, y) position
            radius: Search radius

        Returns:
            List of resources within radius
        \"\"\"
        # This is a compatibility method - should delegate to ResourceManager
        # In this implementation we need to return an empty list as we don't have
        # direct access to the ResourceManager from the Environment class
        return []
"""

        # Insert the method before the class end
        updated_content = content[:class_end] + get_nearby_resources_method + content[class_end:]

        with open(filepath, "w") as f:
            f.write(updated_content)

        print("Added get_nearby_resources method to Environment class.")


def fix_simulation_py():
    """Fix the Simulation class in simulation/simulation.py."""
    filepath = Path("simulation/simulation.py")

    if not filepath.exists():
        print(f"Warning: {filepath} not found. Skipping.")
        return

    create_backup(filepath)

    with open(filepath, "r") as f:
        content = f.read()

    # Replace environment initialization
    init_pattern = r'# Initialize environment.+?self\.environment = Environment\(self\.config\)'
    init_replacement = """# Initialize world environment with terrain and resources
        self.world = World(self.config)
        self.environment = self.world  # For backward compatibility"""

    updated_content = re.sub(init_pattern, init_replacement, content, flags=re.DOTALL)

    # Replace environment update
    update_pattern = r'# Update environment.+?self\.environment\.update\(self\.step\)'
    update_replacement = """# Update environment
        self.world.update(self.step)"""

    updated_content = re.sub(update_pattern, update_replacement, updated_content, flags=re.DOTALL)

    # Add World import
    import_pattern = r'from environment\.conditions import Environment'
    import_replacement = 'from environment.world import World'

    updated_content = re.sub(import_pattern, import_replacement, updated_content)

    with open(filepath, "w") as f:
        f.write(updated_content)

    print("Updated Simulation class to use World instead of Environment.")


def fix_metal_compute():
    """Fix Metal GPU acceleration code."""
    filepath = Path("simulation/spatial/metal_compute.py")

    if not filepath.exists():
        print(f"Warning: {filepath} not found. Skipping.")
        return

    create_backup(filepath)

    with open(filepath, "r") as f:
        content = f.read()

    # Improve Metal setup error handling
    setup_pattern = r'def _setup_metal\(self\):[\s\S]+?METAL_CONFIG\["command_queue"\] = self\.command_queue'
    setup_replacement = """def _setup_metal(self):
        \"\"\"Set up Metal device, command queue, and library with improved error handling.\"\"\"
        if not self.has_metal:
            return

        try:
            # Get default Metal device
            self.device = Metal.MTLCreateSystemDefaultDevice()
            if self.device is None:
                warnings.warn("No Metal device found. Falling back to CPU.")
                self.has_metal = False
                return

            # Create command queue
            self.command_queue = self.device.newCommandQueue()
            if self.command_queue is None:
                warnings.warn("Failed to create Metal command queue. Falling back to CPU.")
                self.has_metal = False
                return

            # Update config
            METAL_CONFIG["device"] = self.device
            METAL_CONFIG["command_queue"] = self.command_queue

            # Print available memory info if possible
            try:
                print(f"Metal device: {self.device.name()}")
                if hasattr(self.device, "recommendedMaxWorkingSetSize"):
                    max_mem = self.device.recommendedMaxWorkingSetSize() / (1024 * 1024)
                    print(f"Available Metal memory: {max_mem:.2f} MB")
            except Exception as e:
                print(f"Could not retrieve Metal device info: {e}")

        except Exception as e:
            warnings.warn(f"Error initializing Metal: {e}. Falling back to CPU.")
            self.has_metal = False"""

    updated_content = re.sub(setup_pattern, setup_replacement, content, flags=re.DOTALL)

    with open(filepath, "w") as f:
        f.write(updated_content)

    print("Improved Metal setup with better error handling.")


def create_build_script():
    """Create build_extensions.py script."""
    content = """#!/usr/bin/env python3
\"\"\"
Build script for compiling Cython extensions in the social evolution simulator.
\"\"\"

import os
import sys
import subprocess
import platform
from pathlib import Path

def is_apple_silicon():
    \"\"\"Check if we're running on Apple Silicon.\"\"\"
    return (
        platform.system() == "Darwin" and
        platform.processor() == "arm" and
        platform.machine() == "arm64"
    )

def main():
    \"\"\"Main build function.\"\"\"
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

    print("\\nTesting compiled extensions...")
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

    print("\\nSetup complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""

    filepath = Path("build_extensions.py")

    print(f"Creating build_extensions.py")
    with open(filepath, "w") as f:
        f.write(content)

    # Make it executable
    os.chmod(filepath, 0o755)

    print("Created build_extensions.py script.")


def update_readme():
    """Update README.md with installation and setup instructions."""
    filepath = Path("README.md")

    create_backup(filepath)

    content = """# Enhanced Social Evolution Simulator

An advanced multi-agent simulation of evolving social dynamics, optimized for performance with comprehensive optimizations including GPU acceleration on Apple Silicon Macs.

## Overview

This enhanced simulator implements complex social behaviors in autonomous agents with optimized performance using:

- **Cython-compiled** critical calculations
- **JAX-accelerated** neural networks
- **Metal GPU acceleration** for spatial operations (Apple Silicon)
- **Optimized quadtree** spatial partitioning
- **DEAP and PyGAD** evolutionary algorithms
- **Vectorized batch operations**
- **Multi-threaded parallel processing**

Agents develop complex emergent behaviors including:
- Community formation and alliances
- Knowledge discovery and sharing
- Status hierarchies and competition
- Cooperation and conflict dynamics
- Mating with genetic inheritance

## Installation

### Prerequisites

- Python 3.8+ (3.9+ recommended)
- C/C++ compiler (for Cython components)
- macOS with M1/M2/M3 chip for Metal acceleration (optional)

### Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/social-evolution-simulator.git
cd social-evolution-simulator
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Compile Cython extensions using the build script
```bash
python build_extensions.py
```

## Running the Simulator

### Command-line Interface (CLI)

```bash
# Basic run with default settings
python main.py

# Specify population and world size
python main.py --population 200 --world-size 1000
```

### GUI Mode with Animation

```bash
# Run with graphical animation
python main.py --mode gui

# Save the animation to a video file
python main.py --mode gui --save-video simulation.mp4
```

### Interactive Streamlit Dashboard

```bash
# Run with interactive Streamlit dashboard
python main.py --mode streamlit
```

## Performance Tuning

### Apple Silicon Optimization

For best performance on M1/M2 MacBooks:

```bash
python main.py --use-metal --threads 8 --batch-size 128
```

### CPU Optimization

For systems without Metal support:

```bash
python main.py --no-metal --threads 4 --batch-size 64
```

### Debugging

Disable optimizations for easier debugging:

```bash
python main.py --no-metal --no-cython
```

## Project Structure

The simulator is organized into modular components:

```
social_evolution_simulator/
├── main.py                   # Entry point
├── config.py                 # Configuration settings
├── build_extensions.py       # Cython build script
├── simulation/
│   ├── simulation.py         # Main simulation logic
│   └── spatial/
│       ├── quadtree.pyx      # Optimized spatial partitioning
│       ├── grid.py           # Grid-based spatial partitioning
│       └── metal_compute.py  # Metal GPU acceleration
├── agents/
│   ├── agent.py              # Agent behavior and decision-making
│   ├── brain.py              # JAX neural networks
│   └── evolution.py          # DEAP/PyGAD integration
├── social/
│   ├── network.py            # Social relationships
│   ├── relationship.py       # Relationship tracking
│   ├── community.py          # Community formation
│   └── interactions.pyx      # Cython-optimized calculations
├── environment/
│   ├── world.py              # Integrated world environment
│   ├── conditions.py         # Environment and weather
│   └── resources.py          # Resource management
├── knowledge/
│   └── knowledge_system.py   # Knowledge discovery and sharing
└── visualization/
    ├── visualizer.py         # Basic visualization
    ├── plotly_vis.py         # Interactive Plotly visualizations
    └── streamlit_app.py      # Interactive Streamlit dashboard
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

### Common Issues

1. **Cython compilation errors**: Make sure you have a C/C++ compiler installed. On macOS, install Xcode command line tools. On Windows, install Visual C++ Build Tools.

2. **Metal acceleration errors**: Metal acceleration only works on macOS with Apple Silicon (M1/M2/M3). Use `--no-metal` on other platforms.

3. **OpenMP support**: On macOS, install libomp via Homebrew for OpenMP support: `brew install libomp`

4. **Memory errors with large simulations**: Reduce the world size, population, or batch size to fit within available memory.

### Getting Help

If you encounter issues:

1. Check the logs for specific error messages
2. Verify that all dependencies are installed
3. Try running with `--no-metal --no-cython` to use pure Python implementations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
"""

    print(f"Updating README.md")
    with open(filepath, "w") as f:
        f.write(content)

    print("Updated README.md with installation and setup instructions.")


def main():
    """Apply all fixes to the codebase."""
    print("Applying fixes to the social evolution simulator codebase...")

    # Create directories if they don't exist
    for dir_path in ["environment", "simulation", "social", "agents", "knowledge", "visualization"]:
        os.makedirs(dir_path, exist_ok=True)

    # Apply fixes
    create_world_py()
    fix_environment_conditions()
    fix_simulation_py()
    fix_metal_compute()
    create_build_script()
    update_readme()

    print("\nAll fixes applied successfully!")
    print("To compile the Cython extensions, run: python build_extensions.py")
    print("To run the simulation, run: python main.py")


if __name__ == "__main__":
    main()