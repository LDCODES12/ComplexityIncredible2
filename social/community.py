"""
Community detection and management for social networks.
Implements algorithms for identifying, tracking, and analyzing social communities.
"""

import os
import sys
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional, Union, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

# Try to import Cython-optimized functions
try:
    from social.interactions import detect_communities

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


class Community:
    """
    Represents a social community of agents.
    Tracks membership, relationships, and community-level properties.
    """

    def __init__(self, community_id, members=None):
        """
        Initialize a community.

        Args:
            community_id: Unique identifier for the community
            members: Set of agent IDs in the community (optional)
        """
        self.id = community_id
        self.members = set(members) if members else set()
        self.formed_at = 0  # Simulation step when community formed

        # Community properties
        self.cohesion = 0.0  # Average internal relationship strength
        self.centrality = {}  # Centrality scores for each member
        self.status_hierarchy = {}  # Status scores for each member
        self.core_members = set()  # Members with high centrality
        self.boundary_members = set()  # Members with connections outside community

        # Cultural traits
        self.cultural_traits = {}  # Shared cultural traits
        self.knowledge_pool = set()  # Shared knowledge IDs

        # External relations
        self.allies = {}  # Allied communities: {community_id: strength}
        self.rivals = {}  # Rival communities: {community_id: strength}

    def add_member(self, agent_id):
        """
        Add a member to the community.

        Args:
            agent_id: ID of agent to add

        Returns:
            True if agent was added, False if already a member
        """
        if agent_id in self.members:
            return False

        self.members.add(agent_id)
        return True

    def remove_member(self, agent_id):
        """
        Remove a member from the community.

        Args:
            agent_id: ID of agent to remove

        Returns:
            True if agent was removed, False if not a member
        """
        if agent_id not in self.members:
            return False

        self.members.remove(agent_id)

        # Remove from special sets
        if agent_id in self.core_members:
            self.core_members.remove(agent_id)

        if agent_id in self.boundary_members:
            self.boundary_members.remove(agent_id)

        # Remove from status hierarchy
        if agent_id in self.status_hierarchy:
            del self.status_hierarchy[agent_id]

        return True

    def calculate_cohesion(self, relationships):
        """
        Calculate the cohesion (average internal relationship strength) of the community.

        Args:
            relationships: Dictionary of relationships between agents

        Returns:
            Cohesion score (0-1)
        """
        if len(self.members) <= 1:
            return 0.0

        total_strength = 0.0
        relationship_count = 0

        # Iterate through all possible pairs
        for agent1_id in self.members:
            for agent2_id in self.members:
                if agent1_id >= agent2_id:
                    continue  # Skip self-pairs and duplicates

                # Get relationship key
                key = (min(agent1_id, agent2_id), max(agent1_id, agent2_id))

                # Add relationship strength if exists
                if key in relationships:
                    total_strength += max(0, relationships[key].get_strength())
                    relationship_count += 1

        # Calculate average
        if relationship_count > 0:
            self.cohesion = total_strength / relationship_count
        else:
            self.cohesion = 0.0

        return self.cohesion

    def calculate_centrality(self, relationships):
        """
        Calculate centrality scores for community members.

        Args:
            relationships: Dictionary of relationships between agents

        Returns:
            Dictionary of centrality scores
        """
        if len(self.members) <= 1:
            self.centrality = {member: 1.0 for member in self.members}
            return self.centrality

        # Initialize centrality scores
        self.centrality = {}

        # Calculate degree centrality
        for agent_id in self.members:
            connections = 0
            total_strength = 0.0

            for other_id in self.members:
                if agent_id == other_id:
                    continue

                # Get relationship key
                key = (min(agent_id, other_id), max(agent_id, other_id))

                # Count relationship
                if key in relationships:
                    strength = relationships[key].get_strength()
                    if strength > 0:
                        connections += 1
                        total_strength += strength

            # Calculate centrality score
            if connections > 0:
                # Normalize by maximum possible connections
                max_connections = len(self.members) - 1
                degree_score = connections / max_connections

                # Consider strength
                strength_score = total_strength / connections

                # Combine scores
                self.centrality[agent_id] = (degree_score * 0.7 + strength_score * 0.3)
            else:
                self.centrality[agent_id] = 0.0

        # Identify core members (high centrality)
        self.core_members = {
            agent_id for agent_id, score in self.centrality.items()
            if score > 0.6
        }

        return self.centrality

    def calculate_status_hierarchy(self, agent_statuses):
        """
        Calculate status hierarchy within the community.

        Args:
            agent_statuses: Dictionary or array of agent status values

        Returns:
            Dictionary of normalized status scores
        """
        if not self.members:
            return {}

        # Get status values for community members
        status_values = {}

        for agent_id in self.members:
            if isinstance(agent_statuses, dict):
                if agent_id in agent_statuses:
                    status_values[agent_id] = agent_statuses[agent_id]
            else:
                # Assume it's a numpy array or similar
                if agent_id < len(agent_statuses):
                    status_values[agent_id] = agent_statuses[agent_id]

        # If no status values, set all to equal
        if not status_values:
            self.status_hierarchy = {agent_id: 0.5 for agent_id in self.members}
            return self.status_hierarchy

        # Normalize values to 0-1 range
        min_val = min(status_values.values())
        max_val = max(status_values.values())

        if max_val > min_val:
            for agent_id, value in status_values.items():
                self.status_hierarchy[agent_id] = (value - min_val) / (max_val - min_val)
        else:
            # All values are equal
            for agent_id in status_values:
                self.status_hierarchy[agent_id] = 0.5

        return self.status_hierarchy

    def identify_boundary_members(self, relationships, all_agents):
        """
        Identify members at the boundary with connections outside the community.

        Args:
            relationships: Dictionary of relationships between agents
            all_agents: Set of all agent IDs

        Returns:
            Set of boundary member IDs
        """
        self.boundary_members = set()

        # Find members with connections to non-members
        for agent_id in self.members:
            for other_id in all_agents:
                if other_id in self.members:
                    continue  # Skip members

                # Get relationship key
                key = (min(agent_id, other_id), max(agent_id, other_id))

                # Check if relationship exists
                if key in relationships:
                    strength = relationships[key].get_strength()
                    if abs(strength) > 0.3:  # Significant relationship
                        self.boundary_members.add(agent_id)
                        break

        return self.boundary_members

    def update_cultural_traits(self, agent_traits):
        """
        Update the community's cultural traits based on member traits.

        Args:
            agent_traits: Dictionary of agent traits {agent_id: {trait: value}}

        Returns:
            Dictionary of community cultural traits
        """
        if not self.members:
            return {}

        # Collect all traits
        all_traits = defaultdict(list)

        for agent_id in self.members:
            if agent_id in agent_traits:
                for trait, value in agent_traits[agent_id].items():
                    all_traits[trait].append(value)

        # Calculate average traits
        self.cultural_traits = {}

        for trait, values in all_traits.items():
            if len(values) > len(self.members) / 2:  # Trait is common
                self.cultural_traits[trait] = np.mean(values)

        return self.cultural_traits

    def update_knowledge_pool(self, agent_knowledge):
        """
        Update the community's shared knowledge.

        Args:
            agent_knowledge: Dictionary of agent knowledge {agent_id: set(knowledge_ids)}

        Returns:
            Set of common knowledge IDs
        """
        if not self.members:
            return set()

        # Get knowledge sets for members
        member_knowledge = [
            agent_knowledge.get(agent_id, set())
            for agent_id in self.members
        ]

        if not member_knowledge:
            return set()

        # Find knowledge common to at least half the members
        all_knowledge = set().union(*member_knowledge)
        common_knowledge = set()

        for knowledge_id in all_knowledge:
            count = sum(1 for k_set in member_knowledge if knowledge_id in k_set)
            if count >= len(self.members) / 2:
                common_knowledge.add(knowledge_id)

        self.knowledge_pool = common_knowledge
        return common_knowledge

    def update_external_relations(self, other_communities, relationship_matrix):
        """
        Update relationships with other communities.

        Args:
            other_communities: List of other Community objects
            relationship_matrix: Matrix of relationship strengths

        Returns:
            Tuple of (allies, rivals) dictionaries
        """
        self.allies = {}
        self.rivals = {}

        # Skip if empty community
        if not self.members:
            return self.allies, self.rivals

        # Calculate average relationship with each community
        for other in other_communities:
            if other.id == self.id or not other.members:
                continue

            # Calculate average relationship strength
            total_strength = 0.0
            relationship_count = 0

            for agent_id in self.members:
                for other_agent_id in other.members:
                    if agent_id < len(relationship_matrix) and other_agent_id < len(relationship_matrix):
                        strength = relationship_matrix[agent_id, other_agent_id]
                        total_strength += strength
                        relationship_count += 1

            if relationship_count > 0:
                avg_strength = total_strength / relationship_count

                # Categorize as ally or rival
                if avg_strength > 0.3:
                    self.allies[other.id] = avg_strength
                elif avg_strength < -0.3:
                    self.rivals[other.id] = abs(avg_strength)

        return self.allies, self.rivals

    def calculate_size_ratio(self, total_agents):
        """
        Calculate the community's size ratio compared to total population.

        Args:
            total_agents: Total number of agents

        Returns:
            Size ratio (0-1)
        """
        if total_agents > 0:
            return len(self.members) / total_agents
        return 0.0

    def to_dict(self):
        """
        Convert community to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "members": list(self.members),
            "formed_at": self.formed_at,
            "cohesion": float(self.cohesion),
            "core_members": list(self.core_members),
            "boundary_members": list(self.boundary_members),
            "cultural_traits": {k: float(v) for k, v in self.cultural_traits.items()},
            "knowledge_pool": list(self.knowledge_pool),
            "allies": {int(k): float(v) for k, v in self.allies.items()},
            "rivals": {int(k): float(v) for k, v in self.rivals.items()},
            "status_hierarchy": {int(k): float(v) for k, v in self.status_hierarchy.items()},
            "size": len(self.members)
        }


class CommunityDetector:
    """
    Detects and manages communities within a social network.
    Implements various community detection algorithms.
    """

    def __init__(self, config=None):
        """
        Initialize community detector.

        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or CONFIG
        self.communities = []
        self.community_history = []

    def detect_communities(self, relationship_matrix, threshold=None):
        """
        Detect communities using relationship matrix.

        Args:
            relationship_matrix: Matrix of relationship strengths
            threshold: Minimum relationship strength to consider a connection

        Returns:
            List of communities (sets of agent IDs)
        """
        if threshold is None:
            threshold = self.config["community_threshold"]

        # Use Cython-optimized detection if available
        if HAS_CYTHON:
            communities = detect_communities(relationship_matrix, threshold)
        else:
            communities = self._detect_communities_py(relationship_matrix, threshold)

        return communities

    def _detect_communities_py(self, relationship_matrix, threshold):
        """
        Pure Python implementation of community detection.

        Args:
            relationship_matrix: Matrix of relationship strengths
            threshold: Minimum relationship strength for a connection

        Returns:
            List of communities (sets of agent IDs)
        """
        n = relationship_matrix.shape[0]

        # Create adjacency graph
        graph = defaultdict(set)

        # Add edges for strong relationships
        for i in range(n):
            for j in range(i + 1, n):
                strength = relationship_matrix[i, j]
                if strength >= threshold:
                    graph[i].add(j)
                    graph[j].add(i)

        # Find connected components (communities)
        visited = set()
        communities = []

        for node in range(n):
            if node not in visited:
                community = set()
                queue = [node]

                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        community.add(current)

                        # Add neighbors
                        for neighbor in graph[current]:
                            if neighbor not in visited:
                                queue.append(neighbor)

                if community:
                    communities.append(community)

        return communities

    def update_communities(self, communities, step=0):
        """
        Update community objects based on detected communities.

        Args:
            communities: List of detected communities (sets of agent IDs)
            step: Current simulation step

        Returns:
            List of Community objects
        """
        # Store previous communities
        previous_communities = self.communities

        # Track which old communities continue
        continued = set()

        # Initialize new communities list
        self.communities = []

        # Process each detected community
        for i, members in enumerate(communities):
            if not members:
                continue

            # Check if this matches a previous community
            best_match = -1
            best_overlap = 0

            for j, prev_comm in enumerate(previous_communities):
                if j in continued:
                    continue  # Already continued

                # Calculate overlap
                overlap_size = len(members.intersection(prev_comm.members))
                overlap_ratio = overlap_size / max(1, len(members))

                if overlap_ratio > 0.5 and overlap_size > best_overlap:
                    best_match = j
                    best_overlap = overlap_size

            if best_match >= 0:
                # Update existing community
                community = previous_communities[best_match]
                community.members = members  # Update membership
                continued.add(best_match)
            else:
                # Create new community
                community = Community(len(self.communities), members)
                community.formed_at = step

            self.communities.append(community)

        # Record history if significant change
        if self.communities != previous_communities:
            self.community_history.append((step, [c.to_dict() for c in self.communities]))

        return self.communities

    def analyze_communities(self, relationships, agent_statuses, agent_traits=None, agent_knowledge=None):
        """
        Perform detailed analysis of all communities.

        Args:
            relationships: Dictionary of relationships between agents
            agent_statuses: Status values for agents
            agent_traits: Dictionary of agent traits (optional)
            agent_knowledge: Dictionary of agent knowledge (optional)

        Returns:
            Updated list of Community objects
        """
        # Skip if no communities
        if not self.communities:
            return []

        # Get set of all agents
        all_agents = set()
        for community in self.communities:
            all_agents.update(community.members)

        # Relationship matrix for calculating inter-community relations
        n = max(all_agents) + 1 if all_agents else 0
        relationship_matrix = np.zeros((n, n), dtype=np.float32)

        # Fill relationship matrix
        for (a1, a2), rel in relationships.items():
            if a1 < n and a2 < n:
                strength = rel.get_strength()
                relationship_matrix[a1, a2] = strength
                relationship_matrix[a2, a1] = strength

        # Analyze each community
        for community in self.communities:
            # Calculate internal cohesion
            community.calculate_cohesion(relationships)

            # Calculate member centrality
            community.calculate_centrality(relationships)

            # Calculate status hierarchy
            community.calculate_status_hierarchy(agent_statuses)

            # Identify boundary members
            community.identify_boundary_members(relationships, all_agents)

            # Update cultural traits if provided
            if agent_traits:
                community.update_cultural_traits(agent_traits)

            # Update knowledge pool if provided
            if agent_knowledge:
                community.update_knowledge_pool(agent_knowledge)

            # Update external relations with other communities
            community.update_external_relations(
                [c for c in self.communities if c.id != community.id],
                relationship_matrix
            )

        return self.communities

    def get_community_for_agent(self, agent_id):
        """
        Get the community containing an agent.

        Args:
            agent_id: Agent ID to find

        Returns:
            Tuple of (community_index, community) or (-1, None) if not found
        """
        for i, community in enumerate(self.communities):
            if agent_id in community.members:
                return i, community

        return -1, None

    def calculate_community_statistics(self):
        """
        Calculate overall statistics about communities.

        Returns:
            Dictionary of community statistics
        """
        if not self.communities:
            return {
                "count": 0,
                "avg_size": 0,
                "max_size": 0,
                "avg_cohesion": 0,
                "communities": []
            }

        # Collect statistics
        sizes = [len(c.members) for c in self.communities]
        cohesions = [c.cohesion for c in self.communities]

        stats = {
            "count": len(self.communities),
            "avg_size": float(np.mean(sizes)) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
            "avg_cohesion": float(np.mean(cohesions)) if cohesions else 0,
            "communities": [c.to_dict() for c in self.communities]
        }

        return stats