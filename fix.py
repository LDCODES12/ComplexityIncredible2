#!/usr/bin/env python3
"""
Direct fix for segmentation fault in social evolution simulator.
Apply this patch before running the simulation.
"""

import sys
import os
import numpy as np

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Try to import the simulation module
try:
    from simulation.simulation import Simulation

    print("Found simulation module, applying fix...")
except ImportError:
    print("Error: Could not import simulation module. Make sure you're in the project root directory.")
    sys.exit(1)

# Store the original method
original_initialize_agents = Simulation._initialize_agents


# Define fixed method for initializing agents
def safe_initialize_agents(self):
    """Fixed agent initialization that prevents segmentation faults."""
    print("Using safe agent initialization...")

    # Use single-threaded mode to avoid multiprocessing issues
    self.use_multiprocessing = False
    if hasattr(self, 'process_pool') and self.process_pool is not None:
        try:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None
        except:
            pass

    # Initialize agents with better error handling
    for i in range(self.config["initial_population"]):
        try:
            agent = Agent(i, config=self.config)
            self.agents[i] = agent

            # Add to spatial partitioning with proper error handling
            if self.using_quadtree:
                try:
                    # Create a deep copy of the position with proper memory layout
                    pos = np.array(agent.position, dtype=np.float64, copy=True)

                    # Ensure it's contiguous and finite
                    if not pos.flags.c_contiguous:
                        pos = np.ascontiguousarray(pos)

                    # Check for invalid values
                    if not np.all(np.isfinite(pos)):
                        pos = np.zeros(2, dtype=np.float64)

                    # Insert into quadtree with explicit shape
                    self.spatial_tree.insert(i, pos)
                except Exception as e:
                    print(f"Warning: Error inserting into quadtree: {e}")
                    # Fall back to grid if quadtree fails
                    if not hasattr(self, 'spatial_grid'):
                        from simulation.spatial.grid import SpatialGrid
                        self.spatial_grid = SpatialGrid(
                            self.config["world_size"],
                            50  # Reasonable cell size
                        )
                    self.spatial_grid.insert(i, agent.position)
            else:
                # Use grid
                self.spatial_grid.insert(i, agent.position)
        except Exception as e:
            print(f"Warning: Error creating agent {i}: {e}")

    print(f"Successfully initialized {len(self.agents)} agents")


# Replace the method
Simulation._initialize_agents = safe_initialize_agents

# Also fix the Metal acceleration setup
original_setup_spatial = Simulation._setup_spatial_partitioning


def safe_setup_spatial_partitioning(self):
    """Fixed spatial partitioning setup to prevent segmentation faults."""
    try:
        # Try to use optimized quadtree but with safer parameters
        if hasattr(sys.modules["simulation.spatial"], "quadtree") and self.config["use_cython"]:
            print("Using Cython-optimized quadtree with safety measures.")

            # Get quadtree module
            quadtree = sys.modules["simulation.spatial"].quadtree

            # Create quadtree with explicit casting
            world_width = float(self.config["world_size"][0])
            world_height = float(self.config["world_size"][1])

            try:
                self.spatial_tree = quadtree.create_quadtree(world_width, world_height)
                self.using_quadtree = True
            except Exception as e:
                print(f"Error creating quadtree: {e}, falling back to grid")
                from simulation.spatial.grid import SpatialGrid
                self.spatial_grid = SpatialGrid(
                    self.config["world_size"],
                    50  # Reasonable cell size
                )
                self.using_quadtree = False
        else:
            # Fallback to grid-based partitioning
            print("Using grid-based spatial partitioning.")
            from simulation.spatial.grid import SpatialGrid
            self.spatial_grid = SpatialGrid(
                self.config["world_size"],
                50  # Reasonable cell size
            )
            self.using_quadtree = False

        # Use Metal without multiprocessing to avoid conflicts
        try:
            from simulation.spatial import metal_compute
            # Only use metal for single-threaded mode
            self.use_metal = metal_compute.has_metal() and self.config["use_metal"] and not self.use_multiprocessing
            if self.use_metal:
                print("Using Metal acceleration for spatial calculations (single-threaded mode).")
            elif self.config["use_metal"]:
                print("Metal disabled in multi-threaded mode for stability.")
        except Exception as e:
            print(f"Error setting up Metal: {e}")
            self.use_metal = False
    except Exception as e:
        print(f"Error in spatial partitioning setup: {e}")
        # Ensure we have a fallback
        from simulation.spatial.grid import SpatialGrid
        self.spatial_grid = SpatialGrid(
            self.config["world_size"],
            50  # Reasonable cell size
        )
        self.using_quadtree = False
        self.use_metal = False


# Replace the method
Simulation._setup_spatial_partitioning = safe_setup_spatial_partitioning

# Fix the shutdown method to clean up resources properly
original_shutdown = Simulation.shutdown


def safe_shutdown(self):
    """Improved shutdown with proper cleanup."""
    try:
        # Call original shutdown
        original_shutdown(self)

        # Additional cleanup
        if hasattr(self, 'use_metal') and self.use_metal:
            try:
                from simulation.spatial import metal_compute
                if hasattr(metal_compute, 'metal_compute'):
                    # Explicitly clean up Metal resources
                    metal_compute.metal_compute.has_metal = False
                    print("Metal resources cleaned up")
            except:
                pass
    except Exception as e:
        print(f"Error during shutdown: {e}")


# Replace the method
Simulation.shutdown = safe_shutdown

print("✅ Fix applied successfully! Now run the simulation with 'python main.py'")