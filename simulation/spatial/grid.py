"""
Grid-based spatial partitioning system.
Provides efficient spatial queries for large numbers of entities.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CONFIG

import numba
from numba import jit, njit


@njit
def calculate_cell_index(position, cell_size, grid_width):
    """
    Calculate the linear index of a grid cell from a position.

    Args:
        position: (x, y) position
        cell_size: Size of each grid cell
        grid_width: Width of the grid in cells

    Returns:
        Linear index of the cell
    """
    cell_x = int(position[0] // cell_size)
    cell_y = int(position[1] // cell_size)
    return cell_y * grid_width + cell_x


@njit
def calculate_cell_coords(position, cell_size):
    """
    Calculate the grid cell coordinates from a position.

    Args:
        position: (x, y) position
        cell_size: Size of each grid cell

    Returns:
        (cell_x, cell_y) grid cell coordinates
    """
    cell_x = int(position[0] // cell_size)
    cell_y = int(position[1] // cell_size)
    return cell_x, cell_y


@njit
def distance(pos1, pos2):
    """Calculate Euclidean distance between two positions."""
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


class SpatialGrid:
    """
    Grid-based spatial partitioning for efficient proximity queries.

    This is a simpler alternative to the quadtree, providing O(1) cell lookups
    at the cost of potentially examining more entities than necessary.
    """

    def __init__(self, world_size, cell_size):
        """
        Initialize spatial grid.

        Args:
            world_size: (width, height) size of the world
            cell_size: Size of each grid cell (or cell side length)
        """
        self.world_size = world_size
        self.cell_size = cell_size

        # Calculate grid dimensions
        self.grid_width = int(np.ceil(world_size[0] / cell_size))
        self.grid_height = int(np.ceil(world_size[1] / cell_size))
        self.grid_size = (self.grid_width, self.grid_height)

        # Initialize empty grid
        # Using a dictionary for sparse representation
        self.grid = {}

        # Track entity positions for updates
        self.entity_positions = {}

    def insert(self, entity_id, position):
        """
        Insert entity into the grid.

        Args:
            entity_id: Unique identifier for the entity
            position: (x, y) position of the entity

        Returns:
            True if insert was successful
        """
        # Calculate cell coordinates
        cell_x, cell_y = calculate_cell_coords(position, self.cell_size)

        # Ensure cell is within grid bounds
        if 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height:
            cell_index = cell_y * self.grid_width + cell_x

            # Add entity to cell
            if cell_index not in self.grid:
                self.grid[cell_index] = set()

            self.grid[cell_index].add(entity_id)

            # Store entity position for updates
            self.entity_positions[entity_id] = position

            return True

        return False

    def remove(self, entity_id, position=None):
        """
        Remove entity from the grid.

        Args:
            entity_id: Unique identifier for the entity
            position: (x, y) position (optional, will use stored position if None)

        Returns:
            True if removal was successful
        """
        # Use provided position or stored position
        if position is None:
            if entity_id in self.entity_positions:
                position = self.entity_positions[entity_id]
            else:
                return False

        # Calculate cell coordinates
        cell_x, cell_y = calculate_cell_coords(position, self.cell_size)

        # Ensure cell is within grid bounds
        if 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height:
            cell_index = cell_y * self.grid_width + cell_x

            # Remove entity from cell
            if cell_index in self.grid and entity_id in self.grid[cell_index]:
                self.grid[cell_index].remove(entity_id)

                # Clean up empty cells
                if not self.grid[cell_index]:
                    del self.grid[cell_index]

                # Remove stored position
                if entity_id in self.entity_positions:
                    del self.entity_positions[entity_id]

                return True

        return False

    def update(self, entity_id, old_position, new_position):
        """
        Update entity position in the grid.

        Args:
            entity_id: Unique identifier for the entity
            old_position: Previous (x, y) position
            new_position: New (x, y) position

        Returns:
            True if update was successful
        """
        # Calculate old and new cell coordinates
        old_cell_x, old_cell_y = calculate_cell_coords(old_position, self.cell_size)
        new_cell_x, new_cell_y = calculate_cell_coords(new_position, self.cell_size)

        # If cell hasn't changed, no update needed
        if old_cell_x == new_cell_x and old_cell_y == new_cell_y:
            # Update stored position
            self.entity_positions[entity_id] = new_position
            return True

        # Remove from old cell
        old_cell_index = old_cell_y * self.grid_width + old_cell_x
        if (0 <= old_cell_x < self.grid_width and
                0 <= old_cell_y < self.grid_height and
                old_cell_index in self.grid and
                entity_id in self.grid[old_cell_index]):

            self.grid[old_cell_index].remove(entity_id)

            # Clean up empty cells
            if not self.grid[old_cell_index]:
                del self.grid[old_cell_index]

        # Insert into new cell
        if 0 <= new_cell_x < self.grid_width and 0 <= new_cell_y < self.grid_height:
            new_cell_index = new_cell_y * self.grid_width + new_cell_x

            if new_cell_index not in self.grid:
                self.grid[new_cell_index] = set()

            self.grid[new_cell_index].add(entity_id)

            # Update stored position
            self.entity_positions[entity_id] = new_position

            return True

        return False

    def get_cell_indices(self, position, radius=1):
        """
        Get grid cell indices within a radius of cells from position.

        Args:
            position: (x, y) position
            radius: Radius in cells (default=1)

        Returns:
            List of cell indices
        """
        center_x, center_y = calculate_cell_coords(position, self.cell_size)

        # Calculate cell range
        min_x = max(0, center_x - radius)
        max_x = min(self.grid_width - 1, center_x + radius)
        min_y = max(0, center_y - radius)
        max_y = min(self.grid_height - 1, center_y + radius)

        # Collect all cell indices
        cell_indices = []
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                cell_indices.append(y * self.grid_width + x)

        return cell_indices

    def get_nearby_entities(self, position, radius):
        """
        Get entities within a radius of position.

        Args:
            position: (x, y) position
            radius: Search radius in world units

        Returns:
            List of (entity_id, position) tuples
        """
        # Calculate cell radius (cells to check)
        cell_radius = int(np.ceil(radius / self.cell_size))

        # Get all cell indices within radius
        cell_indices = self.get_cell_indices(position, cell_radius)

        # Collect all entities in these cells
        entities = []
        for cell_index in cell_indices:
            if cell_index in self.grid:
                for entity_id in self.grid[cell_index]:
                    if entity_id in self.entity_positions:
                        entity_pos = self.entity_positions[entity_id]
                        # Check actual distance
                        if distance(position, entity_pos) <= radius:
                            entities.append((entity_id, entity_pos))

        return entities

    def query_radius(self, position, radius):
        """
        Query entities within a radius (alias for get_nearby_entities).

        Args:
            position: (x, y) position
            radius: Search radius

        Returns:
            List of (entity_id, position) tuples
        """
        return self.get_nearby_entities(position, radius)

    def update_from_metal(self, grid_counts, grid_indices):
        """
        Update the grid using results from Metal GPU computation.

        Args:
            grid_counts: Array of entity counts per cell
            grid_indices: Array of entity indices per cell

        Returns:
            True if update was successful
        """
        # Reset the grid
        self.grid = {}

        # Update with new data
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell_index = y * self.grid_width + x
                count = grid_counts[x, y]

                if count > 0:
                    # Create cell if needed
                    if cell_index not in self.grid:
                        self.grid[cell_index] = set()

                    # Add entities
                    for i in range(count):
                        entity_id = int(grid_indices[x, y, i])
                        self.grid[cell_index].add(entity_id)

        return True

    def clear(self):
        """Clear the grid, removing all entities."""
        self.grid = {}
        self.entity_positions = {}