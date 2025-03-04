"""
Knowledge system for discovery, learning, and sharing information.
Manages knowledge entities, discovery mechanics, and interactions with agents.
"""

import os
import sys
import numpy as np
import random
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional, Union, Any
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG


class Knowledge:
    """
    Represents a piece of knowledge that agents can discover and share.
    Can be a fact, technology, skill, or concept.
    """

    def __init__(self, knowledge_id, name=None, category=None, complexity=None, value=None, prerequisites=None):
        """
        Initialize knowledge.

        Args:
            knowledge_id: Unique identifier
            name: Name of the knowledge (optional)
            category: Category or domain (e.g., "resources", "social", "technology")
            complexity: How difficult it is to learn (0-1 scale)
            value: Benefit/value to an agent that knows it
            prerequisites: List of knowledge IDs required to learn this
        """
        self.id = knowledge_id
        self.name = name or f"Knowledge-{knowledge_id}"
        self.category = category or "general"
        self.complexity = complexity if complexity is not None else random.uniform(0.1, 1.0)
        self.value = value if value is not None else self.complexity * 1.5  # More complex = more valuable
        self.prerequisites = prerequisites or []

        # Discovery tracking
        self.discovered = False
        self.discovered_by = None
        self.discovered_at = None
        self.known_by = set()  # Set of agent IDs that know this knowledge

        # Relations to other knowledge
        self.related_knowledge = {}  # {knowledge_id: similarity_score}

        # Properties that affect agent behavior
        self.effects = {}

    def add_effect(self, effect_type, value):
        """
        Add an effect this knowledge has when learned.

        Args:
            effect_type: Type of effect (e.g., "speed", "energy", "perception")
            value: Magnitude of effect

        Returns:
            Self for chaining
        """
        self.effects[effect_type] = value
        return self

    def get_effects(self):
        """
        Get all effects of this knowledge.

        Returns:
            Dictionary of effects
        """
        return self.effects.copy()

    def discover(self, agent_id, step):
        """
        Mark this knowledge as discovered by an agent.

        Args:
            agent_id: ID of discovering agent
            step: Simulation step when discovered

        Returns:
            True if newly discovered, False if already discovered
        """
        if self.discovered:
            # Already discovered, but mark as known by this agent
            self.known_by.add(agent_id)
            return False

        # Mark as discovered
        self.discovered = True
        self.discovered_by = agent_id
        self.discovered_at = step
        self.known_by.add(agent_id)

        return True

    def can_learn(self, agent_knowledge):
        """
        Check if an agent can learn this knowledge.

        Args:
            agent_knowledge: Set of knowledge IDs the agent already knows

        Returns:
            True if prerequisites are met
        """
        # Check prerequisites
        for prereq_id in self.prerequisites:
            if prereq_id not in agent_knowledge:
                return False

        return True

    def add_relation(self, knowledge_id, similarity):
        """
        Add a relation to another piece of knowledge.

        Args:
            knowledge_id: ID of related knowledge
            similarity: Similarity score (0-1)

        Returns:
            Self for chaining
        """
        self.related_knowledge[knowledge_id] = similarity
        return self

    def get_learning_difficulty(self, agent_learning_ability):
        """
        Calculate how difficult it is for an agent to learn this.

        Args:
            agent_learning_ability: Agent's learning ability (0-1)

        Returns:
            Difficulty score (higher = more difficult)
        """
        # Base difficulty from complexity
        difficulty = self.complexity

        # Adjust based on agent's learning ability
        difficulty /= max(0.1, agent_learning_ability)

        return difficulty

    def to_dict(self):
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "complexity": float(self.complexity),
            "value": float(self.value),
            "prerequisites": self.prerequisites,
            "discovered": self.discovered,
            "discovered_by": self.discovered_by,
            "discovered_at": self.discovered_at,
            "known_by_count": len(self.known_by),
            "effects": self.effects
        }


class KnowledgePool:
    """
    Manages all knowledge in the simulation.
    Handles creation, discovery, and relationships between knowledge.
    """

    def __init__(self, num_knowledge=100, seed=None):
        """
        Initialize knowledge pool.

        Args:
            num_knowledge: Number of knowledge items to generate
            seed: Random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        # Knowledge storage
        self.knowledge = {}
        self.categories = ["resources", "social", "environment", "technology", "health", "strategy"]

        # Knowledge graph for relationships
        self.knowledge_graph = nx.Graph()

        # Discovery statistics
        self.discovery_stats = {
            "total_discovered": 0,
            "discovery_rate": [],
            "discoveries_by_category": defaultdict(int),
            "discovery_timeline": []
        }

        # Generate initial knowledge
        self.generate_knowledge(num_knowledge)

    def generate_knowledge(self, count):
        """
        Generate a set of knowledge items.

        Args:
            count: Number of knowledge items to generate

        Returns:
            Dictionary of generated knowledge
        """
        # Create knowledge items
        for i in range(count):
            category = random.choice(self.categories)
            complexity = random.uniform(0.1, 1.0)
            value = complexity * random.uniform(1.0, 2.0)  # More complex = more valuable

            # Create knowledge
            knowledge = Knowledge(
                i,
                f"{category.capitalize()}-{i}",
                category,
                complexity,
                value
            )

            # Add some effects based on category
            if category == "resources":
                knowledge.add_effect("resource_efficiency", random.uniform(0.1, 0.3))
            elif category == "social":
                knowledge.add_effect("social_influence", random.uniform(0.1, 0.3))
            elif category == "environment":
                knowledge.add_effect("movement_speed", random.uniform(0.1, 0.2))
            elif category == "technology":
                knowledge.add_effect("tool_efficiency", random.uniform(0.1, 0.4))
            elif category == "health":
                knowledge.add_effect("energy_recovery", random.uniform(0.1, 0.3))
            elif category == "strategy":
                knowledge.add_effect("planning_ability", random.uniform(0.1, 0.3))

            # Store knowledge
            self.knowledge[i] = knowledge

            # Add to knowledge graph
            self.knowledge_graph.add_node(i, category=category, complexity=complexity)

        # Create knowledge relationships and prerequisites
        self._create_knowledge_relationships()

        return self.knowledge

    def _create_knowledge_relationships(self):
        """Create relationships between knowledge items."""
        # Sort knowledge by complexity
        sorted_knowledge = sorted(self.knowledge.values(), key=lambda k: k.complexity)

        # Create relationships - similar knowledge is related
        for i, k1 in enumerate(sorted_knowledge):
            # Add some prerequisites (only to more complex knowledge)
            if k1.complexity > 0.3 and random.random() < 0.3:
                # Choose 1-2 prerequisites from less complex knowledge
                available_prereqs = [k2.id for k2 in sorted_knowledge[:i] if k2.complexity < k1.complexity]
                if available_prereqs:
                    num_prereqs = min(len(available_prereqs), random.randint(1, 2))
                    prereqs = random.sample(available_prereqs, num_prereqs)
                    k1.prerequisites = prereqs

            # Create relationships with similar knowledge
            for k2 in sorted_knowledge:
                if k1.id != k2.id and k1.category == k2.category:
                    # Calculate similarity based on complexity difference
                    complexity_diff = abs(k1.complexity - k2.complexity)
                    similarity = max(0.0, 1.0 - complexity_diff)

                    # Add relation if significant
                    if similarity > 0.4:
                        k1.add_relation(k2.id, similarity)
                        self.knowledge_graph.add_edge(k1.id, k2.id, weight=similarity)

    def get_knowledge(self, knowledge_id):
        """
        Get a specific piece of knowledge.

        Args:
            knowledge_id: ID of knowledge to get

        Returns:
            Knowledge object or None
        """
        return self.knowledge.get(knowledge_id)

    def get_random_knowledge(self, excludes=None):
        """
        Get a random piece of knowledge.

        Args:
            excludes: Set of knowledge IDs to exclude

        Returns:
            Random knowledge object or None
        """
        if excludes is None:
            excludes = set()

        available = [k for k in self.knowledge.keys() if k not in excludes]
        if not available:
            return None

        return self.knowledge[random.choice(available)]

    def discover(self, knowledge_id, agent_id, step):
        """
        Mark knowledge as discovered by an agent.

        Args:
            knowledge_id: ID of knowledge
            agent_id: ID of discovering agent
            step: Simulation step when discovered

        Returns:
            True if discovery was successful
        """
        if knowledge_id in self.knowledge:
            knowledge = self.knowledge[knowledge_id]

            # Check if this is a new discovery
            new_discovery = not knowledge.discovered

            # Mark as discovered
            knowledge.discover(agent_id, step)

            # Update statistics if newly discovered
            if new_discovery:
                self.discovery_stats["total_discovered"] += 1
                self.discovery_stats["discoveries_by_category"][knowledge.category] += 1
                self.discovery_stats["discovery_timeline"].append({
                    "step": step,
                    "knowledge_id": knowledge_id,
                    "agent_id": agent_id,
                    "category": knowledge.category
                })

            return True

        return False

    def get_related_knowledge(self, knowledge_id, threshold=0.7):
        """
        Get knowledge related to a specific piece.

        Args:
            knowledge_id: ID of knowledge
            threshold: Minimum similarity threshold

        Returns:
            List of related knowledge IDs
        """
        if knowledge_id not in self.knowledge:
            return []

        knowledge = self.knowledge[knowledge_id]
        related = [
            k_id for k_id, similarity in knowledge.related_knowledge.items()
            if similarity >= threshold
        ]

        return related

    def get_prerequisites_for(self, knowledge_id):
        """
        Get prerequisites for learning a piece of knowledge.

        Args:
            knowledge_id: ID of knowledge

        Returns:
            List of prerequisite knowledge IDs
        """
        if knowledge_id in self.knowledge:
            return self.knowledge[knowledge_id].prerequisites
        return []

    def get_discovery_rate(self):
        """
        Calculate the current discovery rate.

        Returns:
            Percentage of knowledge discovered
        """
        total = len(self.knowledge)
        discovered = self.discovery_stats["total_discovered"]

        if total > 0:
            return 100.0 * discovered / total
        return 0.0

    def get_category_discovery_rates(self):
        """
        Calculate discovery rates by category.

        Returns:
            Dictionary of {category: percentage_discovered}
        """
        # Count total knowledge by category
        category_counts = defaultdict(int)
        for knowledge in self.knowledge.values():
            category_counts[knowledge.category] += 1

        # Calculate percentage discovered by category
        category_rates = {}
        for category, discovered in self.discovery_stats["discoveries_by_category"].items():
            total = category_counts[category]
            if total > 0:
                category_rates[category] = 100.0 * discovered / total
            else:
                category_rates[category] = 0.0

        return category_rates

    def get_knowledge_tree(self):
        """
        Get a tree representation of knowledge prerequisites.

        Returns:
            Dictionary representing the knowledge tree
        """
        tree = {}

        # Create a directed graph of prerequisites
        prereq_graph = nx.DiGraph()

        for k_id, knowledge in self.knowledge.items():
            prereq_graph.add_node(k_id)
            for prereq_id in knowledge.prerequisites:
                prereq_graph.add_edge(prereq_id, k_id)

        # Find root knowledge (no prerequisites)
        roots = [n for n in prereq_graph.nodes() if prereq_graph.in_degree(n) == 0]

        # Build tree from roots
        for root in roots:
            tree[root] = self._build_subtree(root, prereq_graph)

        return tree

    def _build_subtree(self, node, graph):
        """
        Recursively build a subtree of the knowledge hierarchy.

        Args:
            node: Root node
            graph: Prerequisite graph

        Returns:
            Subtree dictionary
        """
        subtree = {}

        # Get all children (knowledge that has this as prerequisite)
        children = list(graph.successors(node))

        for child in children:
            subtree[child] = self._build_subtree(child, graph)

        return subtree

    def get_discovery_stats(self):
        """
        Get statistics about knowledge discovery.

        Returns:
            Discovery statistics dictionary
        """
        stats = self.discovery_stats.copy()
        stats["discovery_rate"] = self.get_discovery_rate()
        stats["category_rates"] = self.get_category_discovery_rates()
        return stats

    def get_knowledge_clusters(self):
        """
        Get clusters of related knowledge using community detection.

        Returns:
            List of knowledge clusters (lists of knowledge IDs)
        """
        # Use NetworkX community detection
        try:
            from networkx.algorithms import community

            # Find communities using Louvain algorithm
            communities = community.louvain_communities(self.knowledge_graph)

            # Convert to list of lists
            return [list(comm) for comm in communities]
        except (ImportError, AttributeError):
            # Fallback if advanced community detection not available
            clusters = []

            # Simple clustering by category
            by_category = defaultdict(list)
            for k_id, knowledge in self.knowledge.items():
                by_category[knowledge.category].append(k_id)

            for category, k_ids in by_category.items():
                clusters.append(k_ids)

            return clusters


class KnowledgeIndex:
    """
    Efficient index for searching and organizing knowledge.
    Supports semantic search and knowledge discovery.
    """

    def __init__(self, knowledge_pool):
        """
        Initialize knowledge index.

        Args:
            knowledge_pool: KnowledgePool to index
        """
        self.knowledge_pool = knowledge_pool

        # Build indices
        self.category_index = defaultdict(list)
        self.complexity_buckets = defaultdict(list)
        self.prerequisite_map = defaultdict(list)

        self._build_indices()

    def _build_indices(self):
        """Build all indices for efficient search."""
        # Clear existing indices
        self.category_index.clear()
        self.complexity_buckets.clear()
        self.prerequisite_map.clear()

        # Index by category
        for k_id, knowledge in self.knowledge_pool.knowledge.items():
            self.category_index[knowledge.category].append(k_id)

            # Index by complexity (in 0.1 buckets)
            bucket = int(knowledge.complexity * 10)
            self.complexity_buckets[bucket].append(k_id)

            # Index by prerequisites
            for prereq_id in knowledge.prerequisites:
                self.prerequisite_map[prereq_id].append(k_id)

    def search_by_category(self, category):
        """
        Search knowledge by category.

        Args:
            category: Category to search for

        Returns:
            List of knowledge IDs in category
        """
        return self.category_index.get(category, [])

    def search_by_complexity(self, min_complexity, max_complexity):
        """
        Search knowledge by complexity range.

        Args:
            min_complexity: Minimum complexity (0-1)
            max_complexity: Maximum complexity (0-1)

        Returns:
            List of knowledge IDs in complexity range
        """
        min_bucket = max(0, int(min_complexity * 10))
        max_bucket = min(10, int(max_complexity * 10) + 1)

        results = []
        for bucket in range(min_bucket, max_bucket):
            results.extend(self.complexity_buckets.get(bucket, []))

        return results

    def find_learnable_knowledge(self, agent_knowledge):
        """
        Find knowledge that an agent can learn based on prerequisites.

        Args:
            agent_knowledge: Set of knowledge IDs the agent already knows

        Returns:
            List of learnable knowledge IDs
        """
        learnable = []

        # Check each knowledge item
        for k_id, knowledge in self.knowledge_pool.knowledge.items():
            if k_id not in agent_knowledge and knowledge.can_learn(agent_knowledge):
                learnable.append(k_id)

        return learnable

    def find_next_discoveries(self, agent_knowledge):
        """
        Find the most likely next discoveries for an agent.

        Args:
            agent_knowledge: Set of knowledge IDs the agent already knows

        Returns:
            List of (knowledge_id, probability) tuples
        """
        # Start with learnable knowledge
        learnable = self.find_learnable_knowledge(agent_knowledge)

        # Calculate discovery probability for each
        discoveries = []
        for k_id in learnable:
            knowledge = self.knowledge_pool.get_knowledge(k_id)

            # Base probability from complexity (inverse)
            probability = 1.0 - knowledge.complexity

            # Adjust based on related knowledge the agent knows
            related = knowledge.related_knowledge
            known_related = sum(1 for rel_id, sim in related.items() if rel_id in agent_knowledge)

            # Boost probability based on related knowledge
            if related:
                boost = 0.2 * known_related / len(related)
                probability += boost

            # Normalize
            probability = min(0.95, max(0.05, probability))

            discoveries.append((k_id, probability))

        # Sort by probability (highest first)
        discoveries.sort(key=lambda x: x[1], reverse=True)

        return discoveries