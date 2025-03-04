"""
Relationship management module for tracking and updating social connections.
Provides mechanisms for recording interactions and calculating relationship strengths.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

# Try to import Cython-optimized functions
try:
    from social.interactions import calculate_relationship_strength

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


class Relationship:
    """
    Represents a social relationship between two agents.
    Tracks interaction history and calculates relationship metrics.
    """

    def __init__(self, agent1_id, agent2_id, initial_value=0.0, memory_span=None):
        """
        Initialize a new relationship.

        Args:
            agent1_id: ID of first agent
            agent2_id: ID of second agent
            initial_value: Initial relationship value (-1 to 1)
            memory_span: Number of interactions to remember (None = use config)
        """
        # Ensure consistent ordering of agent IDs
        if agent1_id > agent2_id:
            agent1_id, agent2_id = agent2_id, agent1_id

        self.agent1_id = agent1_id
        self.agent2_id = agent2_id
        self.memory_span = memory_span or CONFIG["memory_span"]

        # Relationship properties
        self.history = []
        if initial_value != 0.0:
            self.history.append(initial_value)

        self.trust = 0.5  # Initial trust level (0-1)
        self.knowledge_shared = 0.0  # Amount of knowledge shared
        self.cooperation_count = 0  # Number of cooperative interactions
        self.conflict_count = 0  # Number of conflict interactions
        self.last_interaction_time = 0  # Time of last interaction

    def record_interaction(self, value, time_step=None, perspective=None):
        """
        Record a new interaction between the agents.

        Args:
            value: Interaction value (-1 to 1)
            time_step: Current simulation step (optional)
            perspective: Agent ID that initiated the interaction (optional)

        Returns:
            Updated relationship strength
        """
        # Flip value if perspective is agent2
        if perspective is not None and perspective == self.agent2_id:
            value = -value

        # Add to history
        self.history.append(value)

        # Keep history within memory limits
        if len(self.history) > self.memory_span:
            self.history.pop(0)

        # Update last interaction time
        if time_step is not None:
            self.last_interaction_time = time_step

        # Update interaction counters
        if value > 0.3:
            self.cooperation_count += 1
        elif value < -0.3:
            self.conflict_count += 1

        # Update trust based on consistency
        self._update_trust()

        # Return updated relationship strength
        return self.get_strength()

    def _update_trust(self):
        """Update trust based on interaction consistency."""
        if len(self.history) > 1:
            # Calculate consistency as 1 - normalized standard deviation
            std_dev = np.std(self.history) / 2.0  # Divide by 2 to normalize to range [0-1]
            consistency = 1.0 - min(1.0, std_dev)

            # Update trust (80% previous trust, 20% new consistency)
            self.trust = max(0.0, min(1.0, 0.8 * self.trust + 0.2 * consistency))

    def get_strength(self, recency_weight=0.2):
        """
        Calculate overall relationship strength.

        Args:
            recency_weight: Weight given to recent interactions (0-1)

        Returns:
            Relationship strength (-1 to 1)
        """
        if not self.history:
            return 0.0

        # Use Cython implementation if available
        if HAS_CYTHON:
            return calculate_relationship_strength(np.array(self.history), recency_weight)

        # Otherwise use numpy implementation
        # Apply recency weighting
        weights = np.ones(len(self.history))
        for i in range(len(weights)):
            weights[i] = 1.0 + recency_weight * (i / len(weights))

        # Calculate weighted average
        return float(np.sum(np.array(self.history) * weights) / np.sum(weights))

    def record_knowledge_sharing(self, amount=1.0):
        """
        Record knowledge being shared between agents.

        Args:
            amount: Amount of knowledge shared (default=1.0)

        Returns:
            Updated knowledge shared total
        """
        self.knowledge_shared += amount
        return self.knowledge_shared

    def get_sentiment(self):
        """
        Get qualitative sentiment of the relationship.

        Returns:
            Sentiment category as string
        """
        strength = self.get_strength()

        if strength > 0.7:
            return "strong_positive"
        elif strength > 0.3:
            return "positive"
        elif strength > -0.3:
            return "neutral"
        elif strength > -0.7:
            return "negative"
        else:
            return "strong_negative"

    def get_decay_factor(self, current_time):
        """
        Calculate relationship decay based on time since last interaction.

        Args:
            current_time: Current simulation time step

        Returns:
            Decay factor (0-1) where 1 means no decay
        """
        # Calculate time since last interaction
        time_since_last = current_time - self.last_interaction_time

        # No decay if interacted recently
        if time_since_last <= 10:
            return 1.0

        # Slow decay over time (minimum 0.5 after very long time)
        return max(0.5, 1.0 - 0.01 * min(50, time_since_last - 10))

    def apply_decay(self, current_time):
        """
        Apply time-based decay to the relationship.

        Args:
            current_time: Current simulation time step

        Returns:
            Updated relationship strength
        """
        # Get decay factor
        decay = self.get_decay_factor(current_time)

        # If significant decay, add a decay value to history
        if decay < 0.9 and self.history:
            # Calculate decay value as average * decay
            avg_value = np.mean(self.history)
            decay_value = avg_value * (decay - 1.0)  # Will be negative

            # Add to history
            self.history.append(decay_value)

            # Keep history within memory limits
            if len(self.history) > self.memory_span:
                self.history.pop(0)

        # Return updated strength
        return self.get_strength()

    def merge(self, other_relationship):
        """
        Merge another relationship's history into this one.

        Args:
            other_relationship: Another Relationship object

        Returns:
            Updated relationship strength
        """
        # Verify agents match
        if {self.agent1_id, self.agent2_id} != {other_relationship.agent1_id, other_relationship.agent2_id}:
            raise ValueError("Cannot merge relationships between different agents")

        # Combine histories
        combined_history = self.history + other_relationship.history
        combined_history.sort()  # Assume chronological order

        # Keep only the most recent interactions up to memory_span
        if len(combined_history) > self.memory_span:
            combined_history = combined_history[-self.memory_span:]

        self.history = combined_history

        # Update other metrics
        self.trust = max(self.trust, other_relationship.trust)
        self.knowledge_shared += other_relationship.knowledge_shared
        self.cooperation_count += other_relationship.cooperation_count
        self.conflict_count += other_relationship.conflict_count
        self.last_interaction_time = max(self.last_interaction_time, other_relationship.last_interaction_time)

        # Update trust based on consistency
        self._update_trust()

        return self.get_strength()

    def to_dict(self):
        """
        Convert relationship to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "agent1_id": self.agent1_id,
            "agent2_id": self.agent2_id,
            "history": self.history,
            "trust": float(self.trust),
            "knowledge_shared": float(self.knowledge_shared),
            "cooperation_count": self.cooperation_count,
            "conflict_count": self.conflict_count,
            "last_interaction_time": self.last_interaction_time,
            "strength": float(self.get_strength()),
            "sentiment": self.get_sentiment()
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create a relationship from dictionary data.

        Args:
            data: Dictionary with relationship data

        Returns:
            Relationship object
        """
        relationship = cls(data["agent1_id"], data["agent2_id"])
        relationship.history = data["history"]
        relationship.trust = data["trust"]
        relationship.knowledge_shared = data["knowledge_shared"]
        relationship.cooperation_count = data["cooperation_count"]
        relationship.conflict_count = data["conflict_count"]
        relationship.last_interaction_time = data["last_interaction_time"]

        return relationship


class RelationshipPool:
    """
    Efficient pool for managing and recycling relationship objects.
    """

    def __init__(self, max_size=1000):
        """
        Initialize the relationship pool.

        Args:
            max_size: Maximum pool size
        """
        self.max_size = max_size
        self.available = []
        self.total_created = 0

    def get(self, agent1_id, agent2_id, initial_value=0.0):
        """
        Get a relationship object from the pool.

        Args:
            agent1_id: ID of first agent
            agent2_id: ID of second agent
            initial_value: Initial relationship value

        Returns:
            Relationship object
        """
        if self.available:
            # Reuse an existing relationship
            relationship = self.available.pop()

            # Reset it for new use
            if agent1_id > agent2_id:
                agent1_id, agent2_id = agent2_id, agent1_id

            relationship.agent1_id = agent1_id
            relationship.agent2_id = agent2_id
            relationship.history = [initial_value] if initial_value != 0.0 else []
            relationship.trust = 0.5
            relationship.knowledge_shared = 0.0
            relationship.cooperation_count = 0
            relationship.conflict_count = 0
            relationship.last_interaction_time = 0
        else:
            # Create a new relationship
            relationship = Relationship(agent1_id, agent2_id, initial_value)
            self.total_created += 1

        return relationship

    def release(self, relationship):
        """
        Return a relationship to the pool.

        Args:
            relationship: Relationship object to release
        """
        # Only add to pool if not full
        if len(self.available) < self.max_size:
            self.available.append(relationship)