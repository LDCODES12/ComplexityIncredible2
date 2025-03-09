"""
World environment that integrates terrain, resources, and weather systems.
Manages the overall simulation environment and provides unified access.
"""

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
    """
    Integrated world environment that combines terrain, resources, and weather.
    Provides a unified interface for the simulation to interact with the environment.
    """

    def __init__(self, config=None):
        """
        Initialize the world environment.

        Args:
            config: Configuration dictionary (optional)
        """
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
        """
        Update the world for a new simulation step.

        Args:
            step: Current simulation step

        Returns:
            Updated environmental conditions
        """
        self.step = step

        # Update environment
        conditions = self.environment.update(step)

        # Update resources
        self.resource_manager.update_resources(step)

        # Expose resources directly
        self.resources = self.resource_manager.resources

        return conditions

    def get_nearby_resources(self, position, radius):
        """
        Get resources near a position.

        Args:
            position: (x, y) position
            radius: Search radius

        Returns:
            List of resources within radius
        """
        return self.resource_manager.get_nearby_resources(position, radius)

    def add_resource(self, position=None, value=None, type=None, properties=None):
        """
        Add a new resource to the world.

        Args:
            position: (x, y) position (None = random)
            value: Resource value (None = default)
            type: Resource type (None = random)
            properties: Additional properties (optional)

        Returns:
            Newly created resource
        """
        return self.resource_manager.add_resource(position, value, type, properties)

    def remove_resource(self, resource_id):
        """
        Remove a resource from the world.

        Args:
            resource_id: ID of resource to remove

        Returns:
            True if resource was removed
        """
        return self.resource_manager.remove_resource(resource_id)

    def get_environmental_effects(self):
        """
        Get environmental effects on agents and resources.

        Returns:
            Dictionary of effect modifiers
        """
        return self.environment.get_environment_effects()

    def is_position_valid(self, position):
        """
        Check if a position is valid (within bounds and not blocked).

        Args:
            position: (x, y) position

        Returns:
            True if position is valid
        """
        return self.environment.is_position_valid(position)

    def get_movement_cost(self, position):
        """
        Get the movement cost at a position.

        Args:
            position: (x, y) position

        Returns:
            Movement cost multiplier
        """
        return self.environment.get_movement_cost(position)

    def find_starting_positions(self, num_positions, min_distance=50):
        """
        Find suitable starting positions for agents.

        Args:
            num_positions: Number of positions to find
            min_distance: Minimum distance between positions

        Returns:
            List of (x, y) positions
        """
        return self.environment.terrain.find_starting_positions(num_positions, min_distance)

    def get_rich_resource_area(self, resource_type=None):
        """
        Find an area with high resource potential.

        Args:
            resource_type: Type of resource (optional)

        Returns:
            (x, y) position of rich area
        """
        return self.environment.get_rich_resource_area(resource_type)

    def get_state(self):
        """
        Get the current state of the world.

        Returns:
            Dictionary containing world state
        """
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