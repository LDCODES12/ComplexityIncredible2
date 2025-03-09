"""
Main simulation class that integrates all optimized components.
Manages the overall simulation flow and coordinates between subsystems.
"""

import os
import sys
import time
import random
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Dict, Tuple, Set, Optional, Union, Callable, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

# Import optimized components
try:
    from simulation.spatial import quadtree

    HAS_QUADTREE = True
except ImportError:
    print("Warning: Cython quadtree not available. Using fallback implementation.")
    HAS_QUADTREE = False

from simulation.spatial import metal_compute

# Import simulation components
from agents.agent import Agent
from social.network import SocialNetwork
from environment.world import Environment
from knowledge.knowledge_system import KnowledgePool


class Simulation:
    """
    Manages the entire social evolution simulation with all optimizations.
    Coordinates between agent, social, environmental, and knowledge systems.
    """

    def __init__(self, config=None):
        """
        Initialize simulation with configuration.

        Args:
            config: Custom configuration (optional)
        """
        self.config = config or CONFIG
        self.step = 0
        self.start_time = time.time()

        # Set random seeds for reproducibility
        np.random.seed(self.config["seed"])
        random.seed(self.config["seed"])

        # Initialize statistics tracking
        self.stats = {
            "population": [],
            "avg_energy": [],
            "avg_age": [],
            "knowledge_discovered": [],
            "community_count": [],
            "alliance_count": [],
            "conflict_count": [],
            "cooperation_events": [],
            "innovation_rate": [],
            "resource_efficiency": [],
            "social_connectivity": [],
            "genetic_diversity": [],
            "performance": {
                "update_time": [],
                "agent_time": [],
                "environment_time": [],
                "social_time": [],
                "total_time": []
            }
        }

        # Set up multiprocessing if enabled
        self.use_multiprocessing = self.config["n_threads"] > 1
        if self.use_multiprocessing:
            mp.set_start_method('spawn', force=True)
            self.process_pool = ProcessPoolExecutor(max_workers=self.config["n_threads"])

        # Initialize simulation components
        self.initialize_components()

    def initialize_components(self):
        """Initialize all simulation components."""
        print("Initializing simulation components...")

        # Initialize world environment with terrain and resources
        self.world = World(self.config)
        self.environment = self.world  # For backward compatibility

        # Initialize knowledge pool
        self.knowledge_pool = KnowledgePool(num_knowledge=200)

        # Initialize social network
        self.social_network = SocialNetwork(self.config["max_population"])

        # Set up optimized spatial partitioning
        self._setup_spatial_partitioning()

        # Initialize agents
        self.agents = {}
        self._initialize_agents()

        # Set up batch processing groups
        self._setup_batch_processing()

        print(f"Initialization complete. Created {len(self.agents)} agents.")

    def _setup_spatial_partitioning(self):
        """Set up spatial partitioning system."""
        # First try to use optimized quadtree if available
        if HAS_QUADTREE and self.config["use_cython"]:
            print("Using Cython-optimized quadtree for spatial partitioning.")
            self.spatial_tree = quadtree.create_quadtree(
                self.config["world_size"][0],
                self.config["world_size"][1]
            )
            self.using_quadtree = True
        else:
            # Fallback to grid-based partitioning
            print("Using grid-based spatial partitioning.")
            from simulation.spatial.grid import SpatialGrid
            self.spatial_grid = SpatialGrid(
                self.config["world_size"],
                self.config["spatial_partition_size"]
            )
            self.using_quadtree = False

        # Check if Metal compute is available for acceleration
        self.use_metal = metal_compute.has_metal() and self.config["use_metal"]
        if self.use_metal:
            print("Using Metal acceleration for spatial calculations.")

    def _initialize_agents(self):
        """Initialize the agent population."""
        for i in range(self.config["initial_population"]):
            agent = Agent(i, config=self.config)
            self.agents[i] = agent

            # Add to spatial partitioning
            if self.using_quadtree:
                self.spatial_tree.insert(i, agent.position)
            else:
                self.spatial_grid.insert(i, agent.position)

    def _setup_batch_processing(self):
        """Set up agent batch processing for vectorized operations."""
        # Create agent batches for parallel and vectorized processing
        self.batch_size = self.config["batch_size"]
        self._update_agent_batches()

    def _update_agent_batches(self):
        """Update agent batches for efficient processing."""
        # Group agents into batches for vectorized operations
        agent_ids = list(self.agents.keys())
        random.shuffle(agent_ids)  # Shuffle for better load balancing

        self.agent_batches = []
        for i in range(0, len(agent_ids), self.batch_size):
            batch = agent_ids[i:i + self.batch_size]
            self.agent_batches.append(batch)

    def update(self):
        """
        Update the simulation for one step.

        Returns:
            bool: Whether the simulation should continue
        """
        start_time = time.time()
        self.step += 1

        # Update environment
        self.world.update(self.step)
        env_time = time.time() - env_start

        # Update spatial partitioning if using Metal acceleration
        if self.use_metal and self.step % 5 == 0:
            # Get all agent positions
            positions = np.array([agent.position for agent in self.agents.values()])

            if self.using_quadtree:
                # Rebuild quadtree with Metal-accelerated position calculations
                quadtree.build_quadtree(self.spatial_tree, positions)
            else:
                # Update grid with Metal acceleration
                grid_size = (
                    int(np.ceil(self.config["world_size"][0] / self.config["spatial_partition_size"])),
                    int(np.ceil(self.config["world_size"][1] / self.config["spatial_partition_size"]))
                )
                cell_size = (
                    self.config["spatial_partition_size"],
                    self.config["spatial_partition_size"]
                )

                grid_counts, grid_indices = metal_compute.update_spatial_partition(
                    positions, grid_size, cell_size
                )

                # Update the grid with the results
                self.spatial_grid.update_from_metal(grid_counts, grid_indices)

        # Update agents in batches
        agent_start = time.time()
        alive_agents = {}

        if self.use_multiprocessing:
            # Update agents in parallel
            self._update_agents_parallel()
        else:
            # Update agents sequentially in batches
            self._update_agents_sequential()

        # Filter out dead agents
        alive_agents = {aid: agent for aid, agent in self.agents.items() if agent.alive}
        agent_time = time.time() - agent_start

        # If population dropped too low, add new agents
        if len(alive_agents) < self.config["initial_population"] * 0.5:
            self._repopulate_agents(alive_agents)

        # Replace agents dict with only living agents
        self.agents = alive_agents

        # Update agent batches if population changed significantly
        if abs(len(self.agents) - sum(len(batch) for batch in self.agent_batches)) > 0.1 * len(self.agents):
            self._update_agent_batches()

        # Update social network
        social_start = time.time()
        if self.step % 10 == 0:  # Don't need to update every step
            self.social_network.update(self.agents)

            # Detect communities with Cython-optimized algorithm
            self.social_network.detect_communities()
        social_time = time.time() - social_start

        # Update stats
        self._update_stats()

        # Track performance metrics
        total_time = time.time() - start_time
        self.stats["performance"]["update_time"].append(total_time)
        self.stats["performance"]["agent_time"].append(agent_time)
        self.stats["performance"]["environment_time"].append(env_time)
        self.stats["performance"]["social_time"].append(social_time)
        self.stats["performance"]["total_time"].append(total_time)

        # Print progress update every 100 steps
        if self.step % 100 == 0:
            self._print_progress()

        # Check if simulation should end
        return len(self.agents) > 0 and self.step < self.config["simulation_steps"]

    def _update_agents_sequential(self):
        """Update agents sequentially using batch processing."""
        for batch in self.agent_batches:
            # Get batch of agents
            batch_agents = [self.agents[aid] for aid in batch if aid in self.agents]

            # Skip empty batches
            if not batch_agents:
                continue

            # Vectorized perception for the batch (if supported)
            if self.config["use_jax"]:
                self._batch_process_perceptions(batch_agents)

            # Update each agent
            for agent in batch_agents:
                agent.update(
                    self.environment,
                    self.social_network,
                    self.agents,
                    self.knowledge_pool,
                    self.step,
                    self._get_spatial_query_func()
                )

    def _update_agents_parallel(self):
        """Update agents in parallel using multiprocessing."""
        # Split agents into chunks for parallel processing
        chunks = []
        n_chunks = self.config["n_threads"]
        agent_ids = list(self.agents.keys())
        chunk_size = (len(agent_ids) + n_chunks - 1) // n_chunks

        for i in range(0, len(agent_ids), chunk_size):
            chunk = agent_ids[i:i + chunk_size]
            chunks.append(chunk)

        # Function for updating a chunk of agents
        def update_chunk(chunk_ids):
            updated_agents = {}

            for agent_id in chunk_ids:
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    agent.update(
                        self.environment,
                        self.social_network,
                        self.agents,
                        self.knowledge_pool,
                        self.step,
                        self._get_spatial_query_func()
                    )

                    if agent.alive:
                        updated_agents[agent_id] = agent

            return updated_agents

        # Submit all chunks to the process pool
        futures = [
            self.process_pool.submit(update_chunk, chunk)
            for chunk in chunks
        ]

        # Collect results
        updated_chunks = [future.result() for future in futures]

        # Merge results
        for chunk in updated_chunks:
            for agent_id, agent in chunk.items():
                self.agents[agent_id] = agent

    def _batch_process_perceptions(self, batch_agents):
        """Process perceptions for a batch of agents using JAX vectorization."""
        # This would use JAX to vectorize perception calculations
        # Implementation depends on the specific perception model
        pass  # Placeholder - would be implemented for specific perception model

    def _get_spatial_query_func(self):
        """
        Get the appropriate spatial query function based on the partitioning system.

        Returns:
            Function for querying entities in a radius
        """
        if self.using_quadtree:
            return lambda position, radius: quadtree.quadtree_query_radius(
                self.spatial_tree, position, radius
            )
        else:
            return lambda position, radius: self.spatial_grid.get_nearby_entities(
                position, radius
            )

    def _repopulate_agents(self, alive_agents):
        """Add new agents if population is too low."""
        current_count = len(alive_agents)
        target_count = self.config["initial_population"]

        if current_count < target_count * 0.5:
            # Create new agents to replenish population
            num_to_add = int(target_count * 0.2)  # Add 20% of initial population
            highest_id = max(alive_agents.keys()) if alive_agents else 0

            for i in range(num_to_add):
                new_id = highest_id + i + 1

                # Create new agent
                agent = Agent(new_id, config=self.config)
                alive_agents[new_id] = agent

                # Add to spatial partitioning
                if self.using_quadtree:
                    self.spatial_tree.insert(new_id, agent.position)
                else:
                    self.spatial_grid.insert(new_id, agent.position)

    def _update_stats(self):
        """Update simulation statistics."""
        if len(self.agents) > 0:
            # Basic stats
            self.stats["population"].append(len(self.agents))
            self.stats["avg_energy"].append(np.mean([a.energy for a in self.agents.values()]))
            self.stats["avg_age"].append(np.mean([a.age for a in self.agents.values()]))

            # Knowledge stats
            total_knowledge = sum(len(a.knowledge) for a in self.agents.values())
            self.stats["knowledge_discovered"].append(total_knowledge)

            # Social stats
            self.stats["community_count"].append(len(self.social_network.communities))
            self.stats["alliance_count"].append(len(self.social_network.alliances))

            # Track genetic diversity
            if len(self.agents) >= 2:
                genomes = [list(a.genome.values()) for a in self.agents.values()]
                diversity = np.mean([np.std(trait) for trait in zip(*genomes)])
                self.stats["genetic_diversity"].append(diversity)
            else:
                self.stats["genetic_diversity"].append(0.0)

            # Additional events tracked during simulation
            self.stats["conflict_count"].append(self.social_network.conflict_count)
            self.stats["cooperation_events"].append(self.social_network.cooperation_count)

            # Reset event counters for next iteration
            self.social_network.conflict_count = 0
            self.social_network.cooperation_count = 0

    def _print_progress(self):
        """Print simulation progress."""
        elapsed = time.time() - self.start_time
        population = len(self.agents)
        communities = len(self.social_network.communities)
        steps_per_sec = self.step / elapsed if elapsed > 0 else 0

        print(f"Step {self.step}/{self.config['simulation_steps']} | "
              f"Population: {population} | "
              f"Communities: {communities} | "
              f"Speed: {steps_per_sec:.1f} steps/sec")

    def get_state(self):
        """
        Get current simulation state for visualization or analysis.

        Returns:
            Dict containing simulation state
        """
        return {
            "step": self.step,
            "agents": [
                {
                    "id": agent.id,
                    "position": agent.position,
                    "energy": agent.energy,
                    "age": agent.age,
                    "community": self.social_network.get_community(agent.id)[0],
                    "knowledge": len(agent.knowledge),
                    "genome": agent.genome
                }
                for agent in self.agents.values()
            ],
            "resources": [
                {
                    "id": resource.id,
                    "position": resource.position,
                    "value": resource.value,
                }
                for resource in self.environment.resources.values()
            ],
            "communities": self.social_network.communities,
            "alliances": self.social_network.alliances,
            "environment": {
                "conditions": self.environment.conditions,
            }
        }

    def run(self, num_steps=None, callback=None):
        """
        Run simulation for specified number of steps.

        Args:
            num_steps: Number of steps to run (None = use config)
            callback: Function to call after each step with state

        Returns:
            Dict of simulation statistics
        """
        if num_steps is None:
            num_steps = self.config["simulation_steps"]

        print(f"Starting simulation for {num_steps} steps...")
        self.start_time = time.time()

        try:
            # Run with progress bar
            for _ in tqdm(range(num_steps), desc="Simulation Progress"):
                # Update simulation
                continue_sim = self.update()

                # Call callback if provided
                if callback:
                    callback(self.get_state())

                if not continue_sim:
                    print("Simulation ended early.")
                    break

        except KeyboardInterrupt:
            print("Simulation stopped by user.")

        # Clean up resources
        if self.use_multiprocessing:
            self.process_pool.shutdown()

        elapsed = time.time() - self.start_time
        print(f"Simulation completed after {self.step} steps in {elapsed:.1f}s "
              f"({self.step / elapsed:.1f} steps/sec)")

        return self.stats