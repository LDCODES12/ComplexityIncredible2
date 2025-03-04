"""
Configuration settings for the social evolution simulator.
Centralized to make it easy to adjust parameters.
"""

import os
import multiprocessing as mp
import platform

# Check if we're on macOS with Apple Silicon
is_apple_silicon = (
    platform.system() == "Darwin" and
    platform.processor() == "arm" and
    platform.machine() == "arm64"
)

# Try to import Metal for GPU acceleration, fallback if not available
try:
    if is_apple_silicon:
        import Metal
        import Foundation
        HAS_METAL = True
    else:
        HAS_METAL = False
except ImportError:
    HAS_METAL = False
    print("Warning: Metal library not found. Falling back to CPU processing.")

# Global Configuration
CONFIG = {
    # Simulation parameters
    "world_size": (1000, 1000),
    "initial_population": 200,
    "max_population": 2000,
    "simulation_steps": 10000,
    "seed": 42,

    # Agent parameters
    "max_age": 300,
    "vision_range": 50,
    "max_energy": 100,
    "move_energy_cost": 0.1,
    "think_energy_cost": 0.05,
    "reproduction_energy": 70,
    "mutation_rate": 0.05,
    "learning_rate": 0.1,

    # Resource parameters
    "initial_resources": 500,
    "resource_respawn_rate": 0.02,
    "resource_value": 20,
    "resource_cluster_size": 5,

    # Social parameters
    "community_threshold": 0.6,
    "alliance_formation_rate": 0.3,
    "knowledge_value": 1.5,
    "aggression_factor": 0.5,
    "cooperation_benefit": 1.2,
    "memory_span": 20,
    "status_influence": 0.7,

    # Neural network parameters
    "input_size": 20,
    "hidden_size": 16,
    "output_size": 8,
    "learning_rate": 0.01,

    # Evolutionary parameters
    "population_size": 100,
    "generations": 50,
    "elite_size": 5,
    "crossover_prob": 0.7,
    "tournament_size": 3,

    # Technical parameters
    "use_metal": HAS_METAL,
    "n_threads": min(8, mp.cpu_count()),
    "batch_size": 128,
    "spatial_partition_depth": 8,  # For quadtree
    "lod_thresholds": [100, 200, 500],  # Distance thresholds for LOD
    "use_cython": True,
    "use_numba": True,
    "use_jax": True,
    "use_pygad": True,
    "use_deap": True,

    # Memory optimization
    "use_memory_pool": True,
    "pool_sizes": {
        "agent": 500,
        "resource": 1000,
        "relationship": 5000,
    },

    # Visualization parameters
    "update_interval": 5,  # Steps between visualization updates
    "plot_width": 1000,
    "plot_height": 800,
    "max_visible_agents": 500,  # For performance in large simulations
    "render_mode": "adaptive",  # 'full', 'adaptive', or 'minimal'
}

# Environment variables for JAX
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

# Configuration for Metal
if CONFIG["use_metal"]:
    METAL_CONFIG = {
        "device": None,  # Will be set at runtime
        "command_queue": None,  # Will be set at runtime
        "library": None,  # Will be set at runtime
        "max_buffer_size": 1024 * 1024,  # 1MB
        "workgroup_size": 256,
    }
else:
    METAL_CONFIG = None

# Detailed neural network architecture
NN_CONFIG = {
    "layer_sizes": [CONFIG["input_size"], CONFIG["hidden_size"], CONFIG["output_size"]],
    "activation_funcs": ["relu", "sigmoid"],
    "weight_init_range": 0.1,
    "bias_init_range": 0.05,
    "optimizer": "adam",
    "learning_rate": CONFIG["learning_rate"],
}

# DEAP configuration
DEAP_CONFIG = {
    "fitness_weights": (1.0,),  # Maximizing fitness
    "individual_size": sum([l1 * l2 for l1, l2 in zip(NN_CONFIG["layer_sizes"][:-1], NN_CONFIG["layer_sizes"][1:])]),
    "gene_min": -1.0,
    "gene_max": 1.0,
    "mutate_indpb": 0.1,  # Probability of mutating each gene
    "mutate_sigma": 0.2,  # Standard deviation for mutation
}

# PyGAD configuration
PYGAD_CONFIG = {
    "num_generations": CONFIG["generations"],
    "num_parents_mating": CONFIG["elite_size"],
    "initial_population": None,  # Will be set at runtime
    "mutation_percent_genes": 10,
    "mutation_type": "random",
    "parallel_processing": CONFIG["n_threads"] if CONFIG["n_threads"] > 1 else None,
    "random_mutation_min_val": -1.0,
    "random_mutation_max_val": 1.0,
}