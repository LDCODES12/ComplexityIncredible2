# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
"""
Optimized social interaction calculations using Cython.
Handles relationship strength calculations, trust evaluations, and interaction processing.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, fabs

# Define C types for performance
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t IDX_t

# Import OpenMP if available
from cython.parallel import prange


def calculate_relationship_strength(np.ndarray[DTYPE_t, ndim=1] history, DTYPE_t recency_weight=0.2):
    """
    Calculate relationship strength based on interaction history.
    Optimized with Cython.

    Args:
        history: Array of interaction values (-1 to 1)
        recency_weight: Weight for recent interactions

    Returns:
        Relationship strength (-1 to 1)
    """
    cdef int n = history.shape[0]

    if n == 0:
        return 0.0

    cdef np.ndarray[DTYPE_t, ndim=1] weights = np.ones(n, dtype=np.float64)
    cdef int i
    cdef DTYPE_t total_weight = 0.0
    cdef DTYPE_t weighted_sum = 0.0

    # Apply recency weighting
    for i in range(n):
        weights[i] = 1.0 + recency_weight * (i / n)
        total_weight += weights[i]
        weighted_sum += history[i] * weights[i]

    return weighted_sum / total_weight


def evaluate_social_interactions(
    np.ndarray[DTYPE_t, ndim=2] relationship_matrix,
    np.ndarray[DTYPE_t, ndim=2] history_matrix,
    np.ndarray[DTYPE_t, ndim=1] trust_values,
    int memory_span
):
    """
    Evaluate all social interactions and update relationship matrix.
    This is a critical function called frequently.

    Args:
        relationship_matrix: Matrix of relationship strengths
        history_matrix: Matrix of interaction histories
        trust_values: Current trust values
        memory_span: How many interactions to remember

    Returns:
        Updated relationship matrix, trust values
    """
    cdef int n = relationship_matrix.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] updated_matrix = np.copy(relationship_matrix)
    cdef np.ndarray[DTYPE_t, ndim=1] updated_trust = np.copy(trust_values)

    # Use OpenMP for parallelization if available
    cdef int i, j
    cdef DTYPE_t consistency, strength

    for i in prange(n, nogil=True):
        for j in range(i+1, n):
            # Extract history for this pair
            # This would be more complex in practice but simplified here
            if history_matrix[i, j] != 0:
                # Calculate consistency (simplified)
                consistency = 1.0 - fabs(history_matrix[i, j] - relationship_matrix[i, j]) / 2.0

                # Update trust based on consistency
                updated_trust[i] = max(0.0, min(1.0, 0.8 * trust_values[i] + 0.2 * consistency))
                updated_trust[j] = max(0.0, min(1.0, 0.8 * trust_values[j] + 0.2 * consistency))

                # Update relationship based on history and trust
                strength = history_matrix[i, j] * (0.5 + 0.5 * updated_trust[i])
                updated_matrix[i, j] = 0.9 * relationship_matrix[i, j] + 0.1 * strength
                updated_matrix[j, i] = updated_matrix[i, j]  # Symmetric

    return updated_matrix, updated_trust


def detect_communities(
    np.ndarray[DTYPE_t, ndim=2] relationship_matrix,
    DTYPE_t threshold
):
    """
    Detect communities using a fast connected component algorithm.

    Args:
        relationship_matrix: Matrix of relationship strengths
        threshold: Minimum relationship strength to consider a connection

    Returns:
        List of communities (each a set of agent IDs)
    """
    cdef int n = relationship_matrix.shape[0]
    cdef np.ndarray[IDX_t, ndim=1] labels = np.arange(n, dtype=np.int32)
    cdef int i, j, root_i, root_j

    # Union-find algorithm for connected components
    for i in range(n):
        for j in range(i+1, n):
            if relationship_matrix[i, j] >= threshold:
                # Find roots
                root_i = i
                while labels[root_i] != root_i:
                    root_i = labels[root_i]

                root_j = j
                while labels[root_j] != root_j:
                    root_j = labels[root_j]

                # Union
                if root_i != root_j:
                    labels[root_j] = root_i

    # Path compression
    for i in range(n):
        while labels[i] != labels[labels[i]]:
            labels[i] = labels[labels[i]]

    # Collect communities
    cdef dict communities = {}
    for i in range(n):
        if labels[i] not in communities:
            communities[labels[i]] = set()
        communities[labels[i]].add(i)

    # Convert to list of sets
    return list(communities.values())


def calculate_social_distances(
    np.ndarray[DTYPE_t, ndim=2] positions,
    DTYPE_t max_distance
):
    """
    Calculate social distances between all agents within max_distance.

    Args:
        positions: Array of (x, y) positions
        max_distance: Maximum distance to consider

    Returns:
        Sparse distance matrix (i, j, distance)
    """
    cdef int n = positions.shape[0]
    cdef list distances = []
    cdef DTYPE_t dx, dy, dist
    cdef int i, j

    # Calculate distances in parallel
    for i in prange(n, nogil=True):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = sqrt(dx*dx + dy*dy)

            if dist <= max_distance:
                # We can't append to the list with nogil, so we'll do it later
                with gil:
                    distances.append((i, j, dist))

    return distances