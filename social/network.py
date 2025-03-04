"""
Social network management for agent relationships, communities, and alliances.
Leverages Cython-optimized social calculations for performance.
"""

import os
import sys
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional, Union

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

# Import Cython-optimized social calculations if available
try:
    from social.interactions import (
        calculate_relationship_strength,
        evaluate_social_interactions,
        detect_communities,
        calculate_social_distances
    )

    HAS_CYTHON = True
except ImportError:
    print("Warning: Cython-optimized social calculations not available. Using pure Python implementation.")
    HAS_CYTHON = False


class SocialNetwork:
    """
    Manages all social relationships between agents.
    Handles communities, alliances, status hierarchies, and relationship tracking.
    """

    def __init__(self, max_agents=None, config=None):
        """
        Initialize social network.

        Args:
            max_agents: Maximum number of agents to support
            config: Configuration dictionary
        """
        self.config = config or CONFIG
        self.max_agents = max_agents or self.config["max_population"]

        # Sparse representation of relationships
        self.relationships = {}

        # Communities detected in the network
        self.communities = []

        # Alliance networks between communities
        self.alliances = {}

        # Status hierarchy within communities
        self.status = np.zeros(self.max_agents)

        # Matrices for Cython-optimized calculations
        self._setup_matrices()

        # Event counters
        self.conflict_count = 0
        self.cooperation_count = 0

    def _setup_matrices(self):
        """Set up matrices for Cython-optimized calculations."""
        # Initialize matrices for efficient calculation
        # Use sparse matrices to save memory for large agent populations
        self.relationship_matrix = np.zeros((self.max_agents, self.max_agents), dtype=np.float64)
        self.history_matrix = np.zeros((self.max_agents, self.max_agents), dtype=np.float64)
        self.trust_values = np.zeros(self.max_agents, dtype=np.float64)

    def update(self, agents):
        """
        Update the social network based on current agent state.

        Args:
            agents: Dictionary of agents
        """
        # Update relationship matrix with latest values
        self._update_relationship_matrix(agents)

        # If using Cython, evaluate all social interactions
        if HAS_CYTHON and self.config["use_cython"]:
            self.relationship_matrix, self.trust_values = evaluate_social_interactions(
                self.relationship_matrix,
                self.history_matrix,
                self.trust_values,
                self.config["memory_span"]
            )

    def _update_relationship_matrix(self, agents):
        """
        Update relationship matrix from sparse representations.

        Args:
            agents: Dictionary of agent objects
        """
        # Update matrix with latest relationship values
        for (a1, a2), data in self.relationships.items():
            if a1 < self.max_agents and a2 < self.max_agents:
                if a1 in agents and a2 in agents:  # Only update for living agents
                    history = np.array(data["history"])

                    if HAS_CYTHON and self.config["use_cython"]:
                        strength = calculate_relationship_strength(history, 0.2)
                    else:
                        strength = self._calculate_relationship_strength_py(history, 0.2)

                    self.relationship_matrix[a1, a2] = strength
                    self.relationship_matrix[a2, a1] = strength

    def _calculate_relationship_strength_py(self, history, recency_weight=0.2):
        """
        Pure Python implementation of relationship strength calculation.

        Args:
            history: Array of interaction values (-1 to 1)
            recency_weight: Weight for recent interactions

        Returns:
            Relationship strength (-1 to 1)
        """
        if len(history) == 0:
            return 0.0

        # Apply recency weighting
        weights = np.ones(len(history))
        for i in range(len(weights)):
            weights[i] = 1.0 + recency_weight * (i / len(weights))

        # Calculate weighted average
        return np.sum(history * weights) / np.sum(weights)

    def get_relationship(self, agent1_id, agent2_id):
        """
        Get relationship strength between two agents.

        Args:
            agent1_id: ID of first agent
            agent2_id: ID of second agent

        Returns:
            Relationship strength (-1 to 1)
        """
        # Ensure consistent ordering of agent IDs
        if agent1_id > agent2_id:
            agent1_id, agent2_id = agent2_id, agent1_id

        key = (agent1_id, agent2_id)

        # Check if relationship exists in sparse representation
        if key in self.relationships:
            history = self.relationships[key]["history"]

            if HAS_CYTHON and self.config["use_cython"]:
                return calculate_relationship_strength(np.array(history))
            else:
                return self._calculate_relationship_strength_py(np.array(history))

        # Check if relationship exists in matrix
        if agent1_id < self.max_agents and agent2_id < self.max_agents:
            return self.relationship_matrix[agent1_id, agent2_id]

        return 0.0

    def record_interaction(self, agent1_id, agent2_id, value):
        """
        Record an interaction between two agents.

        Args:
            agent1_id: ID of first agent
            agent2_id: ID of second agent
            value: Value of interaction (-1 to 1)

        Returns:
            Updated relationship strength
        """
        # Ensure consistent ordering of agent IDs
        if agent1_id > agent2_id:
            agent1_id, agent2_id = agent2_id, agent1_id
            value = -value  # Flip the value for consistent perspective

        key = (agent1_id, agent2_id)

        # Create relationship if it doesn't exist
        if key not in self.relationships:
            self.relationships[key] = {
                "history": [],
                "trust": 0.0,
                "knowledge_shared": 0.0,
            }

        # Add interaction to history
        history = self.relationships[key]["history"]
        history.append(value)

        # Keep history within memory limits
        if len(history) > self.config["memory_span"]:
            history.pop(0)

        # Update trust based on consistency of interactions
        if len(history) > 1:
            consistency = 1.0 - np.std(history) / 2.0  # Higher consistency = higher trust
            self.relationships[key]["trust"] = max(0.0, min(1.0,
                                                            0.8 * self.relationships[key]["trust"] + 0.2 * consistency))

        # Update history matrix for Cython optimization
        if agent1_id < self.max_agents and agent2_id < self.max_agents:
            self.history_matrix[agent1_id, agent2_id] = value
            self.history_matrix[agent2_id, agent1_id] = -value  # Opposite perspective

        # Track cooperation/conflict events
        if value > 0.5:
            self.cooperation_count += 1
        elif value < -0.5:
            self.conflict_count += 1

        # Calculate updated relationship strength
        if HAS_CYTHON and self.config["use_cython"]:
            strength = calculate_relationship_strength(np.array(history))
        else:
            strength = self._calculate_relationship_strength_py(np.array(history))

        return strength

    def detect_communities(self, min_relationship=None):
        """
        Detect communities using Cython-optimized algorithm.

        Args:
            min_relationship: Minimum relationship strength to consider a connection

        Returns:
            List of communities (sets of agent IDs)
        """
        if min_relationship is None:
            min_relationship = self.config["community_threshold"]

        if HAS_CYTHON and self.config["use_cython"]:
            # Use Cython-optimized community detection
            self.communities = detect_communities(
                self.relationship_matrix,
                min_relationship
            )
        else:
            # Use pure Python implementation
            self.communities = self._detect_communities_py(min_relationship)

        return self.communities

    def _detect_communities_py(self, min_relationship):
        """
        Pure Python implementation of community detection.

        Args:
            min_relationship: Minimum relationship strength for a connection

        Returns:
            List of communities (sets of agent IDs)
        """
        # Create adjacency graph from relationships
        graph = defaultdict(set)

        # Add edges for strong positive relationships
        for (a1, a2), data in self.relationships.items():
            history = np.array(data["history"])
            strength = self._calculate_relationship_strength_py(history)

            if strength >= min_relationship:
                graph[a1].add(a2)
                graph[a2].add(a1)

        # Find connected components (basic communities)
        visited = set()
        communities = []

        for node in graph:
            if node not in visited:
                community = set()
                queue = [node]

                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        community.add(current)
                        queue.extend(graph[current] - visited)

                if community:
                    communities.append(community)

        return communities

    def update_status(self, agent_ids, outcomes):
        """
        Update status hierarchy based on conflict outcomes.

        Args:
            agent_ids: List of agent IDs involved
            outcomes: List of (winner, loser) tuples
        """
        for winner, loser in outcomes:
            if winner < self.max_agents and loser < self.max_agents:
                # Winner gains status, loser loses status
                self.status[winner] = min(1.0, self.status[winner] + 0.1)
                self.status[loser] = max(0.0, self.status[loser] - 0.05)

    def get_community(self, agent_id):
        """
        Get the community containing an agent.

        Args:
            agent_id: ID of agent

        Returns:
            Tuple of (community_id, community)
        """
        for i, community in enumerate(self.communities):
            if agent_id in community:
                return i, community
        return -1, set()

    def form_alliance(self, community1, community2):
        """
        Form an alliance between two communities.

        Args:
            community1: ID of first community
            community2: ID of second community

        Returns:
            Newly formed alliance data
        """
        key = tuple(sorted([community1, community2]))
        alliance = {
            "formed_at": self.step if hasattr(self, 'step') else 0,
            "strength": 0.5,  # Initial alliance strength
            "interactions": [],
            "resource_sharing": 0.0,
            "knowledge_sharing": 0.0,
            "mutual_defense": 0.0
        }

        self.alliances[key] = alliance
        return alliance

    def update_alliance(self, community1, community2, interaction_value):
        """
        Update an alliance based on interactions between communities.

        Args:
            community1: ID of first community
            community2: ID of second community
            interaction_value: Value of interaction (-1 to 1)

        Returns:
            Whether alliance still exists
        """
        key = tuple(sorted([community1, community2]))
        if key in self.alliances:
            alliance = self.alliances[key]
            alliance["interactions"].append(interaction_value)

            # Keep history manageable
            if len(alliance["interactions"]) > self.config["memory_span"]:
                alliance["interactions"].pop(0)

            # Update alliance strength
            alliance["strength"] = np.mean(alliance["interactions"])

            # Remove alliance if it becomes too weak
            if alliance["strength"] < 0.1:
                del self.alliances[key]
                return False
            return True
        return False

    def get_agent_alliances(self, agent_id):
        """
        Get all alliances involving an agent's community.

        Args:
            agent_id: ID of agent

        Returns:
            List of (community_id, alliance_strength) tuples
        """
        community_id, _ = self.get_community(agent_id)
        if community_id == -1:
            return []

        result = []
        for (c1, c2), data in self.alliances.items():
            if community_id in (c1, c2):
                other = c2 if community_id == c1 else c1
                result.append((other, data["strength"]))

        return result

    def calculate_social_distances(self, positions, max_distance):
        """
        Calculate social distances between all agents within max_distance.

        Args:
            positions: Array of agent positions
            max_distance: Maximum distance to consider

        Returns:
            List of (agent1_id, agent2_id, distance) tuples
        """
        if HAS_CYTHON and self.config["use_cython"]:
            return calculate_social_distances(positions, max_distance)
        else:
            return self._calculate_social_distances_py(positions, max_distance)

    def _calculate_social_distances_py(self, positions, max_distance):
        """
        Pure Python implementation of social distance calculation.

        Args:
            positions: Array of agent positions
            max_distance: Maximum distance to consider

        Returns:
            List of (agent1_id, agent2_id, distance) tuples
        """
        n = positions.shape[0]
        distances = []

        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)

                if dist <= max_distance:
                    distances.append((i, j, dist))

        return distances

    def get_social_graph(self):
        """
        Get a graph representation of the social network.

        Returns:
            Dict with nodes, edges, and communities
        """
        nodes = []
        edges = []

        # Create nodes for all agents with communities
        for agent_id in range(self.max_agents):
            if np.any(self.relationship_matrix[agent_id, :] != 0) or np.any(self.relationship_matrix[:, agent_id] != 0):
                community_id, _ = self.get_community(agent_id)
                nodes.append({
                    "id": agent_id,
                    "community": community_id,
                    "status": float(self.status[agent_id])
                })

        # Create edges for all relationships
        for (a1, a2), data in self.relationships.items():
            history = np.array(data["history"])

            if HAS_CYTHON and self.config["use_cython"]:
                strength = calculate_relationship_strength(history)
            else:
                strength = self._calculate_relationship_strength_py(history)

            # Only include significant relationships
            if abs(strength) > 0.2:
                edges.append({
                    "source": a1,
                    "target": a2,
                    "weight": float(strength),
                    "trust": float(data["trust"])
                })

        # Convert communities to list format for serialization
        community_list = [list(community) for community in self.communities]

        # Convert alliances to list format
        alliance_list = [
            {
                "communities": list(key),
                "strength": float(data["strength"])
            }
            for key, data in self.alliances.items()
        ]

        return {
            "nodes": nodes,
            "edges": edges,
            "communities": community_list,
            "alliances": alliance_list
        }