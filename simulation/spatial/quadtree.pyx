# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
"""
Optimized quadtree implementation for spatial partitioning, implemented in Cython.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs

# Define C types for performance
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t IDX_t

# Maximum number of entities in a leaf node before splitting
DEF MAX_ENTITIES = 8
DEF MAX_DEPTH = 8


cdef struct Point:
    DTYPE_t x
    DTYPE_t y


cdef struct Rect:
    DTYPE_t x
    DTYPE_t y
    DTYPE_t width
    DTYPE_t height


cdef inline bint rect_contains_point(Rect rect, Point point) nogil:
    """Check if a rectangle contains a point."""
    return (
        point.x >= rect.x and
        point.y >= rect.y and
        point.x < rect.x + rect.width and
        point.y < rect.y + rect.height
    )


cdef inline bint rect_intersects_rect(Rect a, Rect b) nogil:
    """Check if two rectangles intersect."""
    return (
        a.x < b.x + b.width and
        a.x + a.width > b.x and
        a.y < b.y + b.height and
        a.y + a.height > b.y
    )


cdef inline DTYPE_t point_distance(Point a, Point b) nogil:
    """Calculate distance between two points."""
    cdef DTYPE_t dx = a.x - b.x
    cdef DTYPE_t dy = a.y - b.y
    return sqrt(dx * dx + dy * dy)


cdef class QuadTreeNode:
    """A node in the quadtree."""

    cdef:
        Rect boundary
        int depth
        bint divided
        list entities
        IDX_t[MAX_ENTITIES] entity_ids
        IDX_t num_entities
        QuadTreeNode northwest
        QuadTreeNode northeast
        QuadTreeNode southwest
        QuadTreeNode southeast

    def __cinit__(self, DTYPE_t x, DTYPE_t y, DTYPE_t width, DTYPE_t height, int depth=0):
        """Initialize a quadtree node with a boundary rectangle."""
        self.boundary.x = x
        self.boundary.y = y
        self.boundary.width = width
        self.boundary.height = height
        self.depth = depth
        self.divided = False
        self.num_entities = 0
        self.entities = []  # Python list for easy access

        # Children are initially None
        self.northwest = None
        self.northeast = None
        self.southwest = None
        self.southeast = None

    cdef void subdivide(self):
        """Subdivide this node into four children."""
        cdef DTYPE_t x = self.boundary.x
        cdef DTYPE_t y = self.boundary.y
        cdef DTYPE_t w = self.boundary.width / 2
        cdef DTYPE_t h = self.boundary.height / 2
        cdef int next_depth = self.depth + 1

        self.northwest = QuadTreeNode(x, y, w, h, next_depth)
        self.northeast = QuadTreeNode(x + w, y, w, h, next_depth)
        self.southwest = QuadTreeNode(x, y + h, w, h, next_depth)
        self.southeast = QuadTreeNode(x + w, y + h, w, h, next_depth)

        self.divided = True

        # Redistribute entities to children
        cdef int i
        cdef IDX_t entity_id
        cdef Point point

        for i in range(self.num_entities):
            entity_id = self.entity_ids[i]
            entity = self.entities[i]
            point.x = entity[0]
            point.y = entity[1]

            self._insert_entity_to_children(entity_id, point)

    cdef bint _insert_entity_to_children(self, IDX_t entity_id, Point point):
        """Insert an entity into the appropriate child node."""
        if rect_contains_point(self.northwest.boundary, point):
            return self.northwest.insert(entity_id, point)
        elif rect_contains_point(self.northeast.boundary, point):
            return self.northeast.insert(entity_id, point)
        elif rect_contains_point(self.southwest.boundary, point):
            return self.southwest.insert(entity_id, point)
        elif rect_contains_point(self.southeast.boundary, point):
            return self.southeast.insert(entity_id, point)
        return False

    cdef bint insert(self, IDX_t entity_id, Point point, object entity=None):
        """Insert an entity into the quadtree."""
        # Check if point is within boundary
        if not rect_contains_point(self.boundary, point):
            return False

        # If this node has space and isn't divided, store here
        if self.num_entities < MAX_ENTITIES and not self.divided and self.depth < MAX_DEPTH:
            self.entity_ids[self.num_entities] = entity_id
            if entity is not None:
                self.entities.append(entity)
            self.num_entities += 1
            return True

        # If this node needs to be subdivided
        if not self.divided and self.depth < MAX_DEPTH:
            self.subdivide()

        # If this node is already subdivided, insert into children
        if self.divided:
            return self._insert_entity_to_children(entity_id, point)

        # If we've reached max depth, store here anyway
        if self.depth >= MAX_DEPTH:
            if self.num_entities < MAX_ENTITIES:
                self.entity_ids[self.num_entities] = entity_id
                if entity is not None:
                    self.entities.append(entity)
                self.num_entities += 1
                return True

        return False

    cdef list query_range(self, Rect range_rect):
        """Find all entities within a range rectangle."""
        cdef list found = []

        # Check if range intersects this node's boundary
        if not rect_intersects_rect(self.boundary, range_rect):
            return found

        # Check entities in this node
        cdef int i
        cdef Point point

        for i in range(self.num_entities):
            entity = self.entities[i]
            point.x = entity[0]
            point.y = entity[1]

            if rect_contains_point(range_rect, point):
                found.append((self.entity_ids[i], entity))

        # If this node is divided, check children
        if self.divided:
            found.extend(self.northwest.query_range(range_rect))
            found.extend(self.northeast.query_range(range_rect))
            found.extend(self.southwest.query_range(range_rect))
            found.extend(self.southeast.query_range(range_rect))

        return found

    cdef list query_radius(self, Point center, DTYPE_t radius):
        """Find all entities within a circular radius."""
        cdef list found = []

        # Create a bounding box for the circle
        cdef Rect range_rect
        range_rect.x = center.x - radius
        range_rect.y = center.y - radius
        range_rect.width = radius * 2
        range_rect.height = radius * 2

        # Check if range intersects this node's boundary
        if not rect_intersects_rect(self.boundary, range_rect):
            return found

        # Check entities in this node
        cdef int i
        cdef Point point

        for i in range(self.num_entities):
            entity = self.entities[i]
            point.x = entity[0]
            point.y = entity[1]

            if point_distance(center, point) <= radius:
                found.append((self.entity_ids[i], entity))

        # If this node is divided, check children
        if self.divided:
            found.extend(self.northwest.query_radius(center, radius))
            found.extend(self.northeast.query_radius(center, radius))
            found.extend(self.southwest.query_radius(center, radius))
            found.extend(self.southeast.query_radius(center, radius))

        return found

    cdef list nearest_neighbor(self, Point target, int k=1):
        """Find k nearest neighbors to a target point."""
        cdef list found = []
        cdef list candidates = []
        cdef DTYPE_t dist
        cdef Point point
        cdef int i

        # Add entities from this node as candidates
        for i in range(self.num_entities):
            entity = self.entities[i]
            point.x = entity[0]
            point.y = entity[1]

            dist = point_distance(target, point)
            candidates.append((dist, self.entity_ids[i], entity))

        # If this node is divided, check children
        if self.divided:
            candidates.extend(self.northwest.nearest_neighbor_candidates(target))
            candidates.extend(self.northeast.nearest_neighbor_candidates(target))
            candidates.extend(self.southwest.nearest_neighbor_candidates(target))
            candidates.extend(self.southeast.nearest_neighbor_candidates(target))

        # Sort by distance and take k nearest
        candidates.sort()
        return candidates[:k]

    cdef list nearest_neighbor_candidates(self, Point target):
        """Helper to collect candidates for nearest neighbor search."""
        cdef list candidates = []
        cdef DTYPE_t dist
        cdef Point point
        cdef int i

        # Add entities from this node as candidates
        for i in range(self.num_entities):
            entity = self.entities[i]
            point.x = entity[0]
            point.y = entity[1]

            dist = point_distance(target, point)
            candidates.append((dist, self.entity_ids[i], entity))

        # If this node is divided, check children
        if self.divided:
            candidates.extend(self.northwest.nearest_neighbor_candidates(target))
            candidates.extend(self.northeast.nearest_neighbor_candidates(target))
            candidates.extend(self.southwest.nearest_neighbor_candidates(target))
            candidates.extend(self.southeast.nearest_neighbor_candidates(target))

        return candidates


cdef class QuadTree:
    """Quadtree for efficient spatial queries."""

    cdef:
        QuadTreeNode root
        DTYPE_t width
        DTYPE_t height

    def __cinit__(self, DTYPE_t width, DTYPE_t height):
        """Initialize the quadtree with dimensions."""
        self.width = width
        self.height = height
        self.root = QuadTreeNode(0, 0, width, height)

    def insert(self, IDX_t entity_id, np.ndarray[DTYPE_t, ndim=1] position):
        """Insert an entity with its position into the quadtree."""
        cdef Point point
        point.x = position[0]
        point.y = position[1]

        return self.root.insert(entity_id, point, position)

    def build(self, np.ndarray[DTYPE_t, ndim=2] positions):
        """Build the quadtree from an array of positions."""
        cdef int n = positions.shape[0]
        cdef int i
        cdef Point point

        # Clear and rebuild
        self.root = QuadTreeNode(0, 0, self.width, self.height)

        for i in range(n):
            point.x = positions[i, 0]
            point.y = positions[i, 1]
            self.root.insert(i, point, positions[i])

        return True

    def query_range(self, np.ndarray[DTYPE_t, ndim=1] position, DTYPE_t width, DTYPE_t height):
        """Query entities within a rectangle."""
        cdef Rect range_rect
        range_rect.x = position[0]
        range_rect.y = position[1]
        range_rect.width = width
        range_rect.height = height

        return self.root.query_range(range_rect)

    def query_radius(self, np.ndarray[DTYPE_t, ndim=1] position, DTYPE_t radius):
        """Query entities within a radius."""
        cdef Point center
        center.x = position[0]
        center.y = position[1]

        return self.root.query_radius(center, radius)

    def nearest_neighbors(self, np.ndarray[DTYPE_t, ndim=1] position, int k=1):
        """Find k nearest neighbors to a point."""
        cdef Point target
        target.x = position[0]
        target.y = position[1]

        return self.root.nearest_neighbor(target, k)


# Python-friendly wrapper functions
def create_quadtree(width, height):
    """Create a new quadtree with the specified dimensions."""
    return QuadTree(width, height)

def build_quadtree(quadtree, positions):
    """Build a quadtree from an array of positions."""
    return quadtree.build(positions)

def quadtree_query_radius(quadtree, position, radius):
    """Query entities within a radius using the quadtree."""
    return quadtree.query_radius(position, radius)

def quadtree_nearest_neighbors(quadtree, position, k=1):
    """Find k nearest neighbors using the quadtree."""
    return quadtree.nearest_neighbors(position, k)