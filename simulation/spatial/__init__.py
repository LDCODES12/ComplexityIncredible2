"""
Spatial utilities for efficient proximity queries and movement simulation.
"""

from simulation.spatial.grid import SpatialGrid

try:
    from simulation.spatial.quadtree import create_quadtree, quadtree_query_radius
    HAS_QUADTREE = True
except ImportError:
    HAS_QUADTREE = False

try:
    from simulation.spatial.metal_compute import (
        calculate_distances,
        calculate_resource_influence,
        update_spatial_partition,
        has_metal
    )
    HAS_METAL = has_metal()
except ImportError:
    HAS_METAL = False

__all__ = [
    'SpatialGrid',
    'HAS_QUADTREE',
    'HAS_METAL'
]

if HAS_QUADTREE:
    __all__ += ['create_quadtree', 'quadtree_query_radius']

if HAS_METAL:
    __all__ += ['calculate_distances', 'calculate_resource_influence', 'update_spatial_partition']