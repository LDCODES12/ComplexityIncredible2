"""
Resource management system for the environment.
Handles resource generation, distribution, and lifecycle.
"""

import os
import sys
import numpy as np
import random
from typing import List, Dict, Tuple, Set, Optional, Union, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

import numba
from numba import jit, njit


@njit
def distance(pos1, pos2):
    """Calculate Euclidean distance between positions."""
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


@njit
def normalize_position(pos, world_size):
    """Normalize position to stay within world boundaries."""
    return (
        max(0, min(world_size[0] - 1, pos[0])),
        max(0, min(world_size[1] - 1, pos[1]))
    )


class Resource:
    """
    Represents a resource in the environment.
    Resources can be food, knowledge sources, or special items.
    """

    def __init__(self, resource_id, position, value=None, type="food", properties=None):
        """
        Initialize a resource.

        Args:
            resource_id: Unique identifier for the resource
            position: (x, y) position in the world
            value: Value/energy content of the resource
            type: Type of resource (food, knowledge, special)
            properties: Additional properties (optional)
        """
        self.id = resource_id
        self.position = position
        self.value = value or CONFIG["resource_value"]
        self.type = type
        self.properties = properties or {}

        # Status
        self.discovered = False
        self.last_harvested = 0
        self.depletion_rate = 0.0
        self.growth_rate = 0.0
        self.respawn_timer = 0

        # For cluster identification
        self.cluster_id = -1

    def update(self, step):
        """
        Update resource for a new simulation step.

        Args:
            step: Current simulation step

        Returns:
            Whether resource is still valid
        """
        # If resource is depleting
        if self.depletion_rate > 0:
            self.value -= self.depletion_rate

            # Check if depleted
            if self.value <= 0:
                return False

        # If resource is growing
        if self.growth_rate > 0:
            self.value += self.growth_rate

            # Cap growth if needed
            if "max_value" in self.properties:
                self.value = min(self.value, self.properties["max_value"])

        # If resource is respawning
        if self.value <= 0 and self.respawn_timer > 0:
            self.respawn_timer -= 1

            # Check if ready to respawn
            if self.respawn_timer <= 0:
                self.value = self.properties.get("respawn_value", CONFIG["resource_value"])
                return True

        return self.value > 0

    def harvest(self, amount=None, step=None):
        """
        Harvest resource and get its value.

        Args:
            amount: Amount to harvest (None = all)
            step: Current simulation step

        Returns:
            Value harvested
        """
        if amount is None:
            # Full harvest
            harvested = self.value
            self.value = 0
        else:
            # Partial harvest
            harvested = min(amount, self.value)
            self.value -= harvested

        # Update harvest time
        if step is not None:
            self.last_harvested = step

        # Start respawn timer if applicable
        if self.value <= 0 and "respawn_time" in self.properties:
            self.respawn_timer = self.properties["respawn_time"]

        return harvested

    def get_discovery_difficulty(self):
        """
        Get how difficult the resource is to discover.

        Returns:
            Difficulty value (0-1)
        """
        return self.properties.get("discovery_difficulty", 0.2)

    def to_dict(self):
        """
        Convert resource to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "position": self.position,
            "value": float(self.value),
            "type": self.type,
            "properties": self.properties,
            "discovered": self.discovered,
            "last_harvested": self.last_harvested,
            "cluster_id": self.cluster_id
        }


class ResourcePool:
    """
    Memory-efficient pool for managing resource objects.
    """

    def __init__(self, max_size=1000):
        """
        Initialize the resource pool.

        Args:
            max_size: Maximum pool size
        """
        self.max_size = max_size
        self.available = []
        self.total_created = 0

    def get(self, resource_id, position, value=None, type="food", properties=None):
        """
        Get a resource object from the pool.

        Args:
            resource_id: Unique identifier for the resource
            position: (x, y) position in the world
            value: Value/energy content of the resource
            type: Type of resource (food, knowledge, special)
            properties: Additional properties (optional)

        Returns:
            Resource object
        """
        if self.available:
            # Reuse an existing resource
            resource = self.available.pop()

            # Reset it for new use
            resource.id = resource_id
            resource.position = position
            resource.value = value or CONFIG["resource_value"]
            resource.type = type
            resource.properties = properties or {}
            resource.discovered = False
            resource.last_harvested = 0
            resource.depletion_rate = 0.0
            resource.growth_rate = 0.0
            resource.respawn_timer = 0
            resource.cluster_id = -1
        else:
            # Create a new resource
            resource = Resource(resource_id, position, value, type, properties)
            self.total_created += 1

        return resource

    def release(self, resource):
        """
        Return a resource to the pool.

        Args:
            resource: Resource object to release
        """
        # Only add to pool if not full
        if len(self.available) < self.max_size:
            self.available.append(resource)


class ResourceManager:
    """
    Manages all resources in the environment.
    Handles resource generation, distribution, clusters, and lifecycle.
    """

    def __init__(self, world_size=None, config=None):
        """
        Initialize resource manager.

        Args:
            world_size: (width, height) of the world
            config: Configuration dictionary (optional)
        """
        self.config = config or CONFIG
        self.world_size = world_size or self.config["world_size"]

        # Resource storage
        self.resources = {}
        self.next_resource_id = 0

        # Resource pools by type
        self.resource_pools = {
            "food": ResourcePool(),
            "knowledge": ResourcePool(),
            "special": ResourcePool()
        }

        # Resource clusters
        self.clusters = []

        # Resource density map
        self.density_map = np.zeros(self.world_size, dtype=np.float32)

        # Resource type distribution
        self.type_distribution = {
            "food": 0.8,  # 80% chance for food
            "knowledge": 0.15,  # 15% chance for knowledge
            "special": 0.05  # 5% chance for special
        }

    def generate_initial_resources(self, count=None):
        """
        Generate initial resources in the environment.

        Args:
            count: Number of resources to generate (None = use config)

        Returns:
            Number of resources generated
        """
        count = count or self.config["initial_resources"]

        # Create clusters first
        self._generate_resource_clusters()

        # Generate resources
        for _ in range(count):
            self.add_resource()

        # Update density map
        self._update_density_map()

        return len(self.resources)

    def _generate_resource_clusters(self, num_clusters=10):
        """
        Generate clusters of resources.

        Args:
            num_clusters: Number of clusters to generate

        Returns:
            List of cluster centers
        """
        # Clear existing clusters
        self.clusters = []

        # Generate cluster centers
        for _ in range(num_clusters):
            center = (
                random.uniform(0, self.world_size[0]),
                random.uniform(0, self.world_size[1])
            )

            cluster = {
                "center": center,
                "radius": random.uniform(20, 100),
                "density": random.uniform(0.5, 1.0),
                "resource_type": self._get_random_resource_type(),
                "value_multiplier": random.uniform(0.8, 1.5)
            }

            self.clusters.append(cluster)

        return self.clusters

    def _get_random_resource_type(self):
        """
        Get a random resource type based on distribution.

        Returns:
            Resource type string
        """
        rand = random.random()
        cumulative = 0

        for type, prob in self.type_distribution.items():
            cumulative += prob
            if rand < cumulative:
                return type

        return "food"  # Default

    def add_resource(self, position=None, value=None, type=None, properties=None):
        """
        Add a new resource to the environment.

        Args:
            position: (x, y) position (None = random)
            value: Resource value (None = default or cluster-based)
            type: Resource type (None = random)
            properties: Additional properties (optional)

        Returns:
            Newly created resource
        """
        # Determine resource type
        if type is None:
            type = self._get_random_resource_type()

        # Choose position
        if position is None:
            position = self._choose_resource_position(type)

        # Find nearest cluster
        nearest_cluster = None
        min_distance = float('inf')

        for cluster in self.clusters:
            dist = distance(position, cluster["center"])
            if dist < cluster["radius"] and dist < min_distance:
                nearest_cluster = cluster
                min_distance = dist

        # Determine value based on cluster
        if value is None:
            if nearest_cluster and nearest_cluster["resource_type"] == type:
                # Use cluster value multiplier
                base_value = self.config["resource_value"]
                distance_factor = 1.0 - (min_distance / nearest_cluster["radius"])
                value = base_value * nearest_cluster["value_multiplier"] * (0.8 + 0.4 * distance_factor)
            else:
                # Use default value
                value = self.config["resource_value"]

        # Set up properties based on type
        if properties is None:
            properties = self._get_default_properties(type)

        # Get resource from appropriate pool
        resource = self.resource_pools[type].get(
            self.next_resource_id, position, value, type, properties
        )

        # Set cluster ID if within a cluster
        if nearest_cluster:
            resource.cluster_id = self.clusters.index(nearest_cluster)

        # Add to resources dictionary
        self.resources[self.next_resource_id] = resource
        self.next_resource_id += 1

        # Update density map
        self._add_to_density_map(position, value)

        return resource

    def _choose_resource_position(self, resource_type):
        """
        Choose a position for a new resource.

        Args:
            resource_type: Type of resource

        Returns:
            (x, y) position
        """
        # Check if we should place in a cluster
        cluster_chance = 0.7  # 70% chance to place in a cluster

        if random.random() < cluster_chance and self.clusters:
            # Choose a random cluster that matches the resource type
            matching_clusters = [
                c for c in self.clusters
                if c["resource_type"] == resource_type
            ]

            if matching_clusters:
                cluster = random.choice(matching_clusters)

                # Choose a random position within the cluster
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0, cluster["radius"])

                # Calculate position
                x = cluster["center"][0] + radius * np.cos(angle)
                y = cluster["center"][1] + radius * np.sin(angle)

                # Ensure within world bounds
                return normalize_position((x, y), self.world_size)

        # Random position
        return (
            random.uniform(0, self.world_size[0]),
            random.uniform(0, self.world_size[1])
        )

    def _get_default_properties(self, resource_type):
        """
        Get default properties for a resource type.

        Args:
            resource_type: Type of resource

        Returns:
            Dictionary of properties
        """
        if resource_type == "food":
            return {
                "nutritional_value": random.uniform(0.8, 1.2),
                "respawn_time": random.randint(20, 100),
                "respawn_value": self.config["resource_value"] * random.uniform(0.8, 1.0),
                "discovery_difficulty": 0.2
            }
        elif resource_type == "knowledge":
            return {
                "knowledge_id": random.randint(0, 999),
                "complexity": random.uniform(0.3, 1.0),
                "discovery_difficulty": 0.5,
                "max_value": self.config["resource_value"] * 2
            }
        elif resource_type == "special":
            return {
                "rarity": random.uniform(0.7, 1.0),
                "discovery_difficulty": 0.7,
                "respawn_time": random.randint(100, 500),
                "effect_duration": random.randint(10, 50)
            }

        return {}

    def remove_resource(self, resource_id):
        """
        Remove a resource from the environment.

        Args:
            resource_id: ID of resource to remove

        Returns:
            True if resource was removed
        """
        if resource_id in self.resources:
            resource = self.resources[resource_id]

            # Remove from density map
            self._remove_from_density_map(resource.position, resource.value)

            # Return to pool
            self.resource_pools[resource.type].release(resource)

            # Remove from dictionary
            del self.resources[resource_id]

            return True

        return False

    def update_resources(self, step):
        """
        Update all resources for a new simulation step.

        Args:
            step: Current simulation step

        Returns:
            Number of resources updated
        """
        # Resources to remove
        to_remove = []

        # Update each resource
        for resource_id, resource in self.resources.items():
            if not resource.update(step):
                to_remove.append(resource_id)

        # Remove depleted resources
        for resource_id in to_remove:
            self.remove_resource(resource_id)

        # Check if we need to respawn resources
        if random.random() < self.config["resource_respawn_rate"]:
            self.add_resource()

        # Update density map periodically
        if step % 10 == 0:
            self._update_density_map()

        return len(self.resources)

    def get_nearby_resources(self, position, radius):
        """
        Get resources near a position.

        Args:
            position: (x, y) position
            radius: Search radius

        Returns:
            List of resources within radius
        """
        nearby = []

        for resource in self.resources.values():
            dist = distance(position, resource.position)
            if dist <= radius:
                nearby.append(resource)

        return nearby

    def find_richest_area(self, resource_type=None, area_size=100):
        """
        Find the area with highest resource density.

        Args:
            resource_type: Type of resource to consider (None = all)
            area_size: Size of the area to consider

        Returns:
            (x, y) position of the center of the richest area
        """
        # Create density map for specific resource type if requested
        if resource_type:
            density_map = np.zeros(self.world_size, dtype=np.float32)

            for resource in self.resources.values():
                if resource.type == resource_type:
                    x, y = int(resource.position[0]), int(resource.position[1])
                    if 0 <= x < self.world_size[0] and 0 <= y < self.world_size[1]:
                        density_map[x, y] += resource.value
        else:
            density_map = self.density_map

        # Apply smoothing to find areas (not just points)
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(density_map, sigma=area_size / 5)

        # Find maximum
        max_pos = np.unravel_index(np.argmax(smoothed), smoothed.shape)

        return max_pos

    def _update_density_map(self):
        """
        Update the resource density map.

        Returns:
            Updated density map
        """
        # Reset density map
        self.density_map = np.zeros(self.world_size, dtype=np.float32)

        # Add each resource
        for resource in self.resources.values():
            self._add_to_density_map(resource.position, resource.value)

        return self.density_map

    def _add_to_density_map(self, position, value):
        """
        Add a resource to the density map.

        Args:
            position: (x, y) position
            value: Resource value
        """
        x, y = int(position[0]), int(position[1])

        if 0 <= x < self.world_size[0] and 0 <= y < self.world_size[1]:
            self.density_map[x, y] += value

    def _remove_from_density_map(self, position, value):
        """
        Remove a resource from the density map.

        Args:
            position: (x, y) position
            value: Resource value
        """
        x, y = int(position[0]), int(position[1])

        if 0 <= x < self.world_size[0] and 0 <= y < self.world_size[1]:
            self.density_map[x, y] = max(0, self.density_map[x, y] - value)

    def get_resource_distribution(self):
        """
        Get statistics about resource distribution.

        Returns:
            Dictionary of resource statistics
        """
        if not self.resources:
            return {
                "count": 0,
                "total_value": 0,
                "avg_value": 0,
                "type_counts": {},
                "cluster_counts": {}
            }

        # Count resources by type
        type_counts = {}
        for resource in self.resources.values():
            type_counts[resource.type] = type_counts.get(resource.type, 0) + 1

        # Count resources by cluster
        cluster_counts = {}
        for resource in self.resources.values():
            if resource.cluster_id >= 0:
                cluster_counts[resource.cluster_id] = cluster_counts.get(resource.cluster_id, 0) + 1

        # Calculate total and average value
        total_value = sum(r.value for r in self.resources.values())
        avg_value = total_value / len(self.resources)

        return {
            "count": len(self.resources),
            "total_value": float(total_value),
            "avg_value": float(avg_value),
            "type_counts": type_counts,
            "cluster_counts": cluster_counts
        }