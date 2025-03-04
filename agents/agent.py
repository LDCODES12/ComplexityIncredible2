"""
Optimized agent implementation with neural network decision making,
evolutionary adaptation, and complex social behaviors.
"""

import os
import sys
import numpy as np
import random
from typing import List, Dict, Tuple, Set, Optional, Union, Callable

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

# Import optimized components
from agents.brain import NeuralNetwork, batch_predict
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


class Agent:
    """
    Intelligent agent with neural network decision making and complex social behaviors.
    Optimized for performance with JAX, Numba, and memory efficiency.
    """

    def __init__(self, agent_id, position=None, genome=None, config=None):
        """
        Initialize agent with optional genome.

        Args:
            agent_id: Unique identifier for the agent
            position: Initial position (optional)
            genome: Inherited genome (optional)
            config: Configuration dictionary (optional)
        """
        self.id = agent_id
        self.config = config or CONFIG

        # Physical attributes
        self.position = position
        if position is None:
            self.position = (
                random.uniform(0, self.config["world_size"][0]),
                random.uniform(0, self.config["world_size"][1])
            )

        self.energy = random.uniform(self.config["max_energy"] * 0.5, self.config["max_energy"])
        self.age = 0
        self.alive = True

        # Create or inherit genome
        if genome is None:
            self.create_genome()
        else:
            self.genome = genome.copy()
            # Apply mutations
            for gene in self.genome:
                if random.random() < self.config["mutation_rate"]:
                    self.genome[gene] += random.normalvariate(0, 0.1)
                    # Ensure values stay in reasonable range
                    self.genome[gene] = max(0, min(1, self.genome[gene]))

        # Create neural network for decision making
        self.brain = self.create_brain()

        # Social attributes
        self.known_agents = set()
        self.knowledge = set()
        self.memory = {
            "interactions": [],  # Recent interactions
            "resource_locations": [],  # Remembered resource locations
            "threats": [],  # Remembered threats
        }

        # Status indicators
        self.status = {
            "hunger": 0.0,  # 0=full, 1=starving
            "fear": 0.0,  # 0=calm, 1=terrified
            "curiosity": random.uniform(0.3, 0.8),  # Desire to explore
            "social_need": random.uniform(0.3, 0.8),  # Desire for company
        }

        # Current intentions
        self.intention = {
            "target": None,  # Current target (position, agent, resource)
            "action": None,  # Current action
            "duration": 0,  # How long to maintain this intention
        }

        # Cached perceptions for batch processing
        self.cached_perceptions = None

    def create_genome(self):
        """Create a random genome for the agent."""
        self.genome = {
            # Personality traits
            "aggression": random.uniform(0.1, 0.9),
            "cooperation": random.uniform(0.1, 0.9),
            "curiosity": random.uniform(0.1, 0.9),
            "social": random.uniform(0.1, 0.9),
            "loyalty": random.uniform(0.1, 0.9),
            "risk_tolerance": random.uniform(0.1, 0.9),

            # Physical attributes
            "speed": random.uniform(0.3, 0.8),
            "strength": random.uniform(0.3, 0.8),
            "perception": random.uniform(0.3, 0.8),
            "resilience": random.uniform(0.3, 0.8),

            # Cognitive attributes
            "learning_ability": random.uniform(0.3, 0.8),
            "memory": random.uniform(0.3, 0.8),
            "planning": random.uniform(0.3, 0.8),
            "innovation": random.uniform(0.3, 0.8),
        }

    def create_brain(self):
        """Create neural network for decision making."""
        # Input size: perceptions about environment, self, and others
        input_size = self.config["input_size"]
        # Hidden layer size
        hidden_size = self.config["hidden_size"]
        # Output size: action probabilities
        output_size = self.config["output_size"]

        # Action mapping:
        # 0=move, 1=eat, 2=attack, 3=flee, 4=share, 5=reproduce, 6=explore, 7=rest

        # Create neural network
        return NeuralNetwork([input_size, hidden_size, output_size])

    def perceive(self, environment, social_network, agents, spatial_query_func=None):
        """
        Gather perceptions about the environment and other agents.
        Optimized with Numba and JAX.

        Args:
            environment: Environment object
            social_network: Social network object
            agents: Dictionary of agent objects
            spatial_query_func: Function to query spatial entities (optional)

        Returns:
            Numpy array of perceptions
        """
        perceptions = []

        # Self perceptions
        perceptions.append(self.energy / self.config["max_energy"])  # Energy level
        perceptions.append(self.age / self.config["max_age"])  # Normalized age
        perceptions.append(self.status["hunger"])  # Hunger
        perceptions.append(self.status["fear"])  # Fear
        perceptions.append(self.status["curiosity"])  # Curiosity
        perceptions.append(self.status["social_need"])  # Social need

        # Environmental perceptions
        if spatial_query_func:
            # Use optimized spatial query
            nearby_entities = spatial_query_func(
                self.position,
                self.config["vision_range"] * self.genome["perception"]
            )

            nearby_resources = []
            for entity_id, entity in nearby_entities:
                if isinstance(entity_id, str) and entity_id.startswith('r'):
                    resource_id = int(entity_id[1:])
                    if resource_id in environment.resources:
                        nearby_resources.append(environment.resources[resource_id])
        else:
            # Fallback to environment's get_nearby_resources
            nearby_resources = environment.get_nearby_resources(
                self.position,
                self.config["vision_range"] * self.genome["perception"]
            )

        # Closest resource
        closest_resource = None
        closest_distance = float('inf')
        for resource in nearby_resources:
            d = distance(np.array(self.position), np.array(resource.position))
            if d < closest_distance:
                closest_distance = d
                closest_resource = resource

        if closest_resource is not None:
            # Normalized distance to closest resource
            perceptions.append(closest_distance / self.config["vision_range"])
            # Direction to closest resource (normalized x and y components)
            direction = (
                (closest_resource.position[0] - self.position[0]) / self.config["vision_range"],
                (closest_resource.position[1] - self.position[1]) / self.config["vision_range"]
            )
            perceptions.append(direction[0])
            perceptions.append(direction[1])
        else:
            # No resources visible
            perceptions.extend([1.0, 0.0, 0.0])

        # Environmental conditions
        perceptions.append(environment.conditions["temperature"])
        perceptions.append(environment.conditions["day_night"])
        perceptions.append(environment.conditions["disaster_risk"])

        # Social perceptions
        nearby_agents = []

        if spatial_query_func:
            # Use optimized spatial query for agents
            nearby_entities = spatial_query_func(
                self.position,
                self.config["vision_range"] * self.genome["perception"]
            )

            for entity_id, entity in nearby_entities:
                if isinstance(entity_id, int) and entity_id != self.id:
                    if entity_id in agents and agents[entity_id].alive:
                        d = distance(np.array(self.position), np.array(agents[entity_id].position))
                        nearby_agents.append((agents[entity_id], d))
        else:
            # Fallback to manual distance calculation
            for agent in agents.values():
                if agent.id != self.id and agent.alive:
                    d = distance(np.array(self.position), np.array(agent.position))
                    if d <= self.config["vision_range"] * self.genome["perception"]:
                        nearby_agents.append((agent, d))

        # Sort by distance
        nearby_agents.sort(key=lambda x: x[1])

        # Closest agent
        if nearby_agents:
            closest_agent, closest_distance = nearby_agents[0]

            # Normalized distance to closest agent
            perceptions.append(closest_distance / self.config["vision_range"])

            # Relationship with closest agent
            relationship = social_network.get_relationship(self.id, closest_agent.id)
            perceptions.append(relationship)

            # Relative strength
            strength_ratio = self.genome["strength"] / closest_agent.genome["strength"]
            perceptions.append(min(1.0, strength_ratio))

            # Is closest agent from same community?
            my_community, _ = social_network.get_community(self.id)
            other_community, _ = social_network.get_community(closest_agent.id)
            same_community = my_community != -1 and my_community == other_community
            perceptions.append(1.0 if same_community else 0.0)
        else:
            # No agents visible
            perceptions.extend([1.0, 0.0, 0.5, 0.0])

        # Community status
        my_community, community_members = social_network.get_community(self.id)
        if my_community != -1:
            # Community size
            perceptions.append(len(community_members) / 20.0)  # Normalize

            # My status in community
            perceptions.append(social_network.status[self.id])
        else:
            perceptions.extend([0.0, 0.0])

        # Knowledge level
        perceptions.append(len(self.knowledge) / 50.0)  # Normalize

        # Ensure consistent length
        while len(perceptions) < self.config["input_size"]:
            perceptions.append(0.0)

        # Limit to expected input size
        self.cached_perceptions = np.array(perceptions[:self.config["input_size"]])
        return self.cached_perceptions

    def decide_action(self, perceptions=None):
        """
        Decide on an action based on perceptions.

        Args:
            perceptions: Optional perceptions (uses cached if None)

        Returns:
            Action index
        """
        if perceptions is None:
            perceptions = self.cached_perceptions

        if perceptions is None:
            # Fallback to random decision if no perceptions
            return random.randint(0, self.config["output_size"] - 1)

        # Get action probabilities from neural network
        action_probs = self.brain.predict(perceptions)

        # Apply personality biases
        action_probs[0] *= 1.0 + 0.5 * self.genome["speed"]  # move
        action_probs[1] *= 1.0 + 0.5 * (1.0 - self.status["hunger"])  # eat
        action_probs[2] *= 1.0 + self.genome["aggression"]  # attack
        action_probs[3] *= 1.0 + self.status["fear"]  # flee
        action_probs[4] *= 1.0 + self.genome["cooperation"]  # share
        action_probs[5] *= 1.0 + 0.5 * (self.energy / self.config["max_energy"])  # reproduce
        action_probs[6] *= 1.0 + self.genome["curiosity"]  # explore
        action_probs[7] *= 1.0 + (1.0 - self.energy / self.config["max_energy"])  # rest

        # Normalize to sum to 1
        action_probs = action_probs / np.sum(action_probs)

        # Choose action (stochastically based on probabilities)
        action = np.random.choice(self.config["output_size"], p=action_probs)

        return action

    def update(self, environment, social_network, agents, knowledge_pool, step, spatial_query_func=None):
        """
        Update agent for a new simulation step.

        Args:
            environment: Environment object
            social_network: Social network object
            agents: Dictionary of agent objects
            knowledge_pool: Knowledge pool object
            step: Current simulation step
            spatial_query_func: Function to query spatial entities (optional)

        Returns:
            Whether the agent is still alive
        """
        if not self.alive:
            return False

        # Age and basic energy consumption
        self.age += 1
        self.energy -= self.config["move_energy_cost"] * 0.1  # Basic metabolism

        # Update internal status
        self.status["hunger"] = max(0.0, min(1.0, 1.0 - self.energy / self.config["max_energy"]))
        self.status["curiosity"] = max(0.0, min(1.0, self.status["curiosity"] +
                                                random.uniform(-0.1, 0.1)))
        self.status["social_need"] = max(0.0, min(1.0, self.status["social_need"] +
                                                  random.uniform(-0.1, 0.1)))

        # Death conditions
        if self.energy <= 0 or self.age >= self.config["max_age"]:
            self.alive = False
            return False

        # Perceive environment
        perceptions = self.perceive(environment, social_network, agents, spatial_query_func)

        # Decide on action
        action = self.decide_action(perceptions)

        # Execute action
        self.execute_action(
            action, environment, social_network, agents, knowledge_pool, step, spatial_query_func
        )

        # Return alive status
        return self.alive

    def execute_action(
            self, action, environment, social_network,
            agents, knowledge_pool, step, spatial_query_func=None
    ):
        """
        Execute the chosen action.

        Args:
            action: Action index
            environment: Environment object
            social_network: Social network object
            agents: Dictionary of agent objects
            knowledge_pool: Knowledge pool object
            step: Current simulation step
            spatial_query_func: Function to query spatial entities (optional)
        """
        # Action mapping: 0=move, 1=eat, 2=attack, 3=flee, 4=share, 5=reproduce, 6=explore, 7=rest

        if action == 0:  # Move
            self.move(environment)
        elif action == 1:  # Eat
            self.eat(environment, spatial_query_func)
        elif action == 2:  # Attack
            self.attack(agents, social_network, spatial_query_func)
        elif action == 3:  # Flee
            self.flee(agents, environment, spatial_query_func)
        elif action == 4:  # Share
            self.share(agents, social_network, knowledge_pool, spatial_query_func)
        elif action == 5:  # Reproduce
            self.reproduce(agents, social_network, environment, step, spatial_query_func)
        elif action == 6:  # Explore
            self.explore(environment, knowledge_pool, step)
        elif action == 7:  # Rest
            self.rest()

    def move(self, environment):
        """
        Move in the environment.

        Args:
            environment: Environment object
        """
        # Decide on a direction
        # If we have a target, move towards it
        if self.intention["target"] is not None and self.intention["action"] == "move":
            target_pos = self.intention["target"]
            # Calculate direction to target
            dx = target_pos[0] - self.position[0]
            dy = target_pos[1] - self.position[1]
            # Normalize
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist > 0:
                dx /= dist
                dy /= dist
        else:
            # Random movement
            angle = random.uniform(0, 2 * np.pi)
            dx = np.cos(angle)
            dy = np.sin(angle)

        # Apply speed
        speed = 5.0 * self.genome["speed"]
        dx *= speed
        dy *= speed

        # Update position
        old_position = self.position
        new_position = (
            self.position[0] + dx,
            self.position[1] + dy
        )

        # Ensure position is within world bounds
        new_position = normalize_position(
            np.array(new_position),
            np.array(environment.world_size)
        )
        self.position = new_position

        # Consume energy for movement
        move_distance = distance(np.array(old_position), np.array(new_position))
        self.energy -= self.config["move_energy_cost"] * move_distance

    def eat(self, environment, spatial_query_func=None):
        """
        Consume a nearby resource.

        Args:
            environment: Environment object
            spatial_query_func: Function to query spatial entities (optional)

        Returns:
            Whether resource was consumed
        """
        if spatial_query_func:
            # Use optimized spatial query
            nearby_entities = spatial_query_func(self.position, 5.0)

            nearby_resources = []
            for entity_id, entity in nearby_entities:
                if isinstance(entity_id, str) and entity_id.startswith('r'):
                    resource_id = int(entity_id[1:])
                    if resource_id in environment.resources:
                        nearby_resources.append(environment.resources[resource_id])
        else:
            # Fallback to environment's get_nearby_resources
            nearby_resources = environment.get_nearby_resources(self.position, 5.0)

        if nearby_resources:
            # Choose closest resource
            closest = min(
                nearby_resources,
                key=lambda r: distance(np.array(self.position), np.array(r.position))
            )

            # Consume the resource
            self.energy = min(self.config["max_energy"], self.energy + closest.value)

            # Remove the consumed resource
            environment.remove_resource(closest.id)

            # Remember the location for future reference
            self.memory["resource_locations"].append({
                "position": closest.position,
                "time": self.age,
                "value": closest.value
            })

            # Limit memory size
            if len(self.memory["resource_locations"]) > 10:
                self.memory["resource_locations"].pop(0)

            return True

        return False

    def attack(self, agents, social_network, spatial_query_func=None):
        """
        Attack another agent.

        Args:
            agents: Dictionary of agent objects
            social_network: Social network object
            spatial_query_func: Function to query spatial entities (optional)

        Returns:
            Whether attack was successful
        """
        # Find nearby agents
        nearby_agents = []

        if spatial_query_func:
            # Use optimized spatial query
            nearby_entities = spatial_query_func(self.position, 10.0)

            for entity_id, entity in nearby_entities:
                if isinstance(entity_id, int) and entity_id != self.id:
                    if entity_id in agents and agents[entity_id].alive:
                        # Get relationship and check if attack is warranted
                        relationship = social_network.get_relationship(self.id, entity_id)
                        attack_threshold = 0.0 - 0.5 * relationship

                        if relationship < attack_threshold:
                            d = distance(np.array(self.position), np.array(agents[entity_id].position))
                            nearby_agents.append((agents[entity_id], d))
        else:
            # Fallback to manual search
            for agent in agents.values():
                if agent.id != self.id and agent.alive:
                    d = distance(np.array(self.position), np.array(agent.position))
                    if d <= 10.0:  # Attack range
                        relationship = social_network.get_relationship(self.id, agent.id)
                        # Less likely to attack those with positive relationships
                        attack_threshold = 0.0 - 0.5 * relationship
                        if relationship < attack_threshold:
                            nearby_agents.append((agent, d))

        if not nearby_agents:
            return False

        # Choose closest valid target
        target, _ = min(nearby_agents, key=lambda x: x[1])

        # Determine attack success based on relative strength
        my_strength = self.genome["strength"] * (0.8 + 0.4 * self.energy / self.config["max_energy"])
        target_strength = target.genome["strength"] * (0.8 + 0.4 * target.energy / self.config["max_energy"])

        success_chance = my_strength / (my_strength + target_strength)

        # Energy cost for attacker
        self.energy -= self.config["move_energy_cost"] * 3

        if random.random() < success_chance:
            # Successful attack
            damage = random.uniform(10, 20) * my_strength
            target.energy -= damage

            # Record negative interaction
            social_network.record_interaction(self.id, target.id, -0.8)

            # Update status
            if target.energy <= 0:
                target.alive = False
                # Winner gets part of loser's energy
                energy_gain = min(target.energy * 0.5, self.config["max_energy"] - self.energy)
                self.energy += energy_gain

                # Update social status
                social_network.update_status(
                    [self.id, target.id],
                    [(self.id, target.id)]
                )

                return True
        else:
            # Failed attack
            # Target counterattacks
            damage = random.uniform(5, 10) * target_strength
            self.energy -= damage

            # Record negative interaction
            social_network.record_interaction(self.id, target.id, -0.5)

            # Update social status
            social_network.update_status(
                [self.id, target.id],
                [(target.id, self.id)]
            )

        return False

    def flee(self, agents, environment, spatial_query_func=None):
        """
        Flee from danger.

        Args:
            agents: Dictionary of agent objects
            environment: Environment object
            spatial_query_func: Function to query spatial entities (optional)

        Returns:
            Whether fleeing succeeded
        """
        # Find nearby threats
        threats = []

        if spatial_query_func:
            # Use optimized spatial query
            nearby_entities = spatial_query_func(
                self.position,
                self.config["vision_range"]
            )

            for entity_id, entity in nearby_entities:
                if isinstance(entity_id, int) and entity_id != self.id:
                    if entity_id in agents and agents[entity_id].alive:
                        agent = agents[entity_id]
                        # Consider agents with high aggression and strength as threats
                        threat_level = agent.genome["aggression"] * agent.genome["strength"]
                        if threat_level > 0.6:  # Threshold for considering a threat
                            d = distance(np.array(self.position), np.array(agent.position))
                            threats.append((agent, d, threat_level))
        else:
            # Fallback to manual search
            for agent in agents.values():
                if agent.id != self.id and agent.alive:
                    d = distance(np.array(self.position), np.array(agent.position))
                    if d <= self.config["vision_range"]:
                        # Consider agents with high aggression and strength as threats
                        threat_level = agent.genome["aggression"] * agent.genome["strength"]
                        if threat_level > 0.6:  # Threshold for considering a threat
                            threats.append((agent, d, threat_level))

        if not threats:
            return False

        # Calculate weighted average threat direction
        dx, dy = 0, 0
        for agent, d, threat in threats:
            # Direction from threat to self
            threat_dx = self.position[0] - agent.position[0]
            threat_dy = self.position[1] - agent.position[1]

            # Normalize
            if d > 0:
                threat_dx /= d
                threat_dy /= d

            # Weight by threat level and proximity
            weight = threat / d
            dx += threat_dx * weight
            dy += threat_dy * weight

        # Normalize direction
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude

        # Apply speed (faster when fleeing)
        speed = 8.0 * self.genome["speed"]
        dx *= speed
        dy *= speed

        # Update position
        old_position = self.position
        new_position = (
            self.position[0] + dx,
            self.position[1] + dy
        )

        # Ensure position is within world bounds
        new_position = normalize_position(
            np.array(new_position),
            np.array(environment.world_size)
        )
        self.position = new_position

        # Consume extra energy for rapid movement
        move_distance = distance(np.array(old_position), np.array(new_position))
        self.energy -= self.config["move_energy_cost"] * move_distance * 1.5

        # Remember threats
        for agent, d, threat in threats:
            self.memory["threats"].append({
                "agent_id": agent.id,
                "position": agent.position,
                "time": self.age,
                "threat_level": threat
            })

        # Limit memory size
        if len(self.memory["threats"]) > 10:
            self.memory["threats"] = self.memory["threats"][-10:]

        return True

    def share(self, agents, social_network, knowledge_pool, spatial_query_func=None):
        """
        Share knowledge or resources with another agent.

        Args:
            agents: Dictionary of agent objects
            social_network: Social network object
            knowledge_pool: Knowledge pool object
            spatial_query_func: Function to query spatial entities (optional)

        Returns:
            Whether sharing succeeded
        """
        # Find nearby agents to share with
        nearby_agents = []

        if spatial_query_func:
            # Use optimized spatial query
            nearby_entities = spatial_query_func(self.position, 10.0)

            for entity_id, entity in nearby_entities:
                if isinstance(entity_id, int) and entity_id != self.id:
                    if entity_id in agents and agents[entity_id].alive:
                        d = distance(np.array(self.position), np.array(agents[entity_id].position))
                        nearby_agents.append((agents[entity_id], d))
        else:
            # Fallback to manual search
            for agent in agents.values():
                if agent.id != self.id and agent.alive:
                    d = distance(np.array(self.position), np.array(agent.position))
                    if d <= 10.0:  # Sharing range
                        nearby_agents.append((agent, d))

        if not nearby_agents:
            return False

        # Choose closest agent
        target, _ = min(nearby_agents, key=lambda x: x[1])

        # Add to known agents
        self.known_agents.add(target.id)
        target.known_agents.add(self.id)

        # Decide what to share based on relationship
        relationship = social_network.get_relationship(self.id, target.id)

        if relationship >= 0.3:
            # Share knowledge
            if self.knowledge and random.random() < 0.7:
                # Choose a random piece of knowledge to share
                knowledge_id = random.choice(list(self.knowledge))

                if knowledge_id not in target.knowledge:
                    target.knowledge.add(knowledge_id)

                    # Knowledge receiver learns based on their learning ability
                    knowledge_value = knowledge_pool.get_knowledge(knowledge_id).value
                    learning_bonus = target.genome["learning_ability"] * knowledge_value
                    target.energy = min(
                        self.config["max_energy"],
                        target.energy + self.config["knowledge_value"] * learning_bonus
                    )

                    # Record the knowledge sharing
                    relation_key = (min(self.id, target.id), max(self.id, target.id))
                    if relation_key in social_network.relationships:
                        social_network.relationships[relation_key]["knowledge_shared"] += 1

            # Record positive interaction
            social_network.record_interaction(self.id, target.id, 0.5)

            # Consider alliances between communities
            my_community, _ = social_network.get_community(self.id)
            other_community, _ = social_network.get_community(target.id)

            if my_community != -1 and other_community != -1 and my_community != other_community:
                # Potentially form or strengthen alliance
                if random.random() < self.config["alliance_formation_rate"]:
                    if (my_community, other_community) not in social_network.alliances and \
                            (other_community, my_community) not in social_network.alliances:
                        social_network.form_alliance(my_community, other_community)
                    else:
                        social_network.update_alliance(my_community, other_community, 0.3)

        # Share energy (food) if relationship is strong and we have excess
        if relationship >= 0.6 and self.energy > self.config["max_energy"] * 0.7:
            shared_energy = self.energy * 0.1  # Share 10% of energy
            self.energy -= shared_energy
            target.energy = min(self.config["max_energy"], target.energy + shared_energy)

            # Record very positive interaction
            social_network.record_interaction(self.id, target.id, 0.8)

        return True

    def reproduce(self, agents, social_network, environment, step, spatial_query_func=None):
        """
        Reproduce with another agent.

        Args:
            agents: Dictionary of agent objects
            social_network: Social network object
            environment: Environment object
            step: Current simulation step
            spatial_query_func: Function to query spatial entities (optional)

        Returns:
            Whether reproduction succeeded
        """
        # Check if we have enough energy
        if self.energy < self.config["reproduction_energy"]:
            return False

        # Find potential mates
        potential_mates = []

        if spatial_query_func:
            # Use optimized spatial query
            nearby_entities = spatial_query_func(self.position, 10.0)

            for entity_id, entity in nearby_entities:
                if isinstance(entity_id, int) and entity_id != self.id:
                    if entity_id in agents and agents[entity_id].alive:
                        agent = agents[entity_id]
                        if agent.energy >= self.config["reproduction_energy"]:
                            relationship = social_network.get_relationship(self.id, agent.id)
                            # More likely to reproduce with agents with positive relationships
                            if relationship > 0.0:
                                d = distance(np.array(self.position), np.array(agent.position))
                                potential_mates.append((agent, relationship))
        else:
            # Fallback to manual search
            for agent in agents.values():
                if agent.id != self.id and agent.alive:
                    d = distance(np.array(self.position), np.array(agent.position))
                    if d <= 10.0 and agent.energy >= self.config["reproduction_energy"]:
                        relationship = social_network.get_relationship(self.id, agent.id)
                        # More likely to reproduce with agents with positive relationships
                        if relationship > 0.0:
                            potential_mates.append((agent, relationship))

        if not potential_mates:
            return False

        # Weight by relationship strength
        weights = [max(0.1, rel) for _, rel in potential_mates]
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
            mate = random.choices([a for a, _ in potential_mates], weights=weights, k=1)[0]
        else:
            mate = random.choice([a for a, _ in potential_mates])

        # Create child genome by combining parents' genomes
        child_genome = {}
        for gene in self.genome:
            # 50% chance to inherit from each parent
            if random.random() < 0.5:
                child_genome[gene] = self.genome[gene]
            else:
                child_genome[gene] = mate.genome[gene]

            # Small chance of mutation
            if random.random() < self.config["mutation_rate"]:
                child_genome[gene] += random.normalvariate(0, 0.1)
                child_genome[gene] = max(0, min(1, child_genome[gene]))

        # Consume energy for reproduction
        self.energy -= self.config["reproduction_energy"] * 0.6
        mate.energy -= self.config["reproduction_energy"] * 0.6

        # Create child agent
        child_position = (
            (self.position[0] + mate.position[0]) / 2,
            (self.position[1] + mate.position[1]) / 2
        )

        # Get the next available ID
        child_id = max(agents.keys()) + 1

        # Create the child agent
        child = Agent(child_id, child_position, child_genome, self.config)

        # Add to agents dictionary
        agents[child_id] = child

        # Record the offspring relationship in social network
        social_network.record_interaction(self.id, child.id, 0.9)
        social_network.record_interaction(mate.id, child.id, 0.9)

        # Record the mating as a positive interaction
        social_network.record_interaction(self.id, mate.id, 0.7)

        return True

    def explore(self, environment, knowledge_pool, step):
        """
        Explore the environment to find resources or gain knowledge.

        Args:
            environment: Environment object
            knowledge_pool: Knowledge pool object
            step: Current simulation step

        Returns:
            Whether exploration yielded something valuable
        """
        # Choose a random direction to explore
        angle = random.uniform(0, 2 * np.pi)
        dx = np.cos(angle) * 6.0 * self.genome["speed"]
        dy = np.sin(angle) * 6.0 * self.genome["speed"]

        # Update position
        old_position = self.position
        new_position = (
            self.position[0] + dx,
            self.position[1] + dy
        )

        # Ensure position is within world bounds
        new_position = normalize_position(
            np.array(new_position),
            np.array(environment.world_size)
        )
        self.position = new_position

        # Consume energy for movement
        move_distance = distance(np.array(old_position), np.array(new_position))
        self.energy -= self.config["move_energy_cost"] * move_distance

        # Chance to discover resources
        nearby_resources = environment.get_nearby_resources(self.position, 15.0)
        for resource in nearby_resources:
            # Remember resource location
            self.memory["resource_locations"].append({
                "position": resource.position,
                "time": self.age,
                "value": resource.value
            })

        # Limit memory size
        if len(self.memory["resource_locations"]) > 20:
            self.memory["resource_locations"] = self.memory["resource_locations"][-20:]

        # Chance to gain new knowledge based on curiosity and learning ability
        discovery_chance = self.genome["curiosity"] * self.genome["learning_ability"] * 0.1
        if random.random() < discovery_chance:
            # Discovery!
            new_knowledge = knowledge_pool.get_random_knowledge(excludes=self.knowledge)
            if new_knowledge:
                self.knowledge.add(new_knowledge.id)
                knowledge_pool.discover(new_knowledge.id, self.id, step)

                # Gain energy from valuable knowledge
                knowledge_bonus = self.config["knowledge_value"] * new_knowledge.value
                self.energy = min(self.config["max_energy"], self.energy + knowledge_bonus)

                return True

        return False

    def rest(self):
        """
        Rest to recover energy.

        Returns:
            Always True (rest always succeeds)
        """
        # Recover energy
        recovery = random.uniform(2, 4) * self.genome["resilience"]
        self.energy = min(self.config["max_energy"], self.energy + recovery)

        # Reduce fear
        self.status["fear"] = max(0.0, self.status["fear"] - 0.2)

        return True

    def to_dict(self):
        """
        Convert agent to dictionary for serialization.

        Returns:
            Dict representation of agent
        """
        return {
            "id": self.id,
            "position": self.position,
            "energy": float(self.energy),
            "age": int(self.age),
            "alive": self.alive,
            "genome": {k: float(v) for k, v in self.genome.items()},
            "knowledge": list(self.knowledge),
            "known_agents": list(self.known_agents),
            "status": {k: float(v) for k, v in self.status.items()},
        }