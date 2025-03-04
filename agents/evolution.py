"""
Evolutionary algorithms for agent evolution using DEAP and PyGAD.
Handles population-level evolution of neural networks and social strategies.
"""

import sys
import os
import numpy as np
import random
from typing import List, Dict, Tuple, Callable, Union, Optional

# DEAP imports
import deap
from deap import base, creator, tools, algorithms

# PyGAD imports
import pygad

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG, DEAP_CONFIG, PYGAD_CONFIG
from agents.brain import NeuralNetwork

# Check for Metal GPU support in PyGAD
try:
    if CONFIG["use_metal"]:
        from pygad.torchga.mps import model_weights_as_matrix, model_weights_as_vector

        HAS_PYGAD_METAL = True
    else:
        HAS_PYGAD_METAL = False
except ImportError:
    HAS_PYGAD_METAL = False


class DEAPEvolution:
    """
    Handles evolutionary algorithms using DEAP.
    Used for evolving neural networks and agent genomes.
    """

    def __init__(self, individual_size=None, fitness_function=None):
        """
        Initialize the DEAP evolution system.

        Args:
            individual_size: Size of the genome (if None, use from config)
            fitness_function: Function to evaluate fitness
        """
        self.individual_size = individual_size or DEAP_CONFIG["individual_size"]
        self.fitness_function = fitness_function

        # Initialize DEAP
        self._setup_deap()

    def _setup_deap(self):
        """Set up DEAP framework with fitness, individuals, and operators."""
        # Create fitness and individual types if they don't exist
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=DEAP_CONFIG["fitness_weights"])

        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        # Set up toolbox
        self.toolbox = base.Toolbox()

        # Register gene and individual creation
        self.toolbox.register(
            "attr_float",
            random.uniform,
            DEAP_CONFIG["gene_min"],
            DEAP_CONFIG["gene_max"]
        )

        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_float,
            n=self.individual_size
        )

        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )

        # Register genetic operators
        self.toolbox.register(
            "mate",
            tools.cxBlend,
            alpha=0.5
        )

        self.toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=0,
            sigma=DEAP_CONFIG["mutate_sigma"],
            indpb=DEAP_CONFIG["mutate_indpb"]
        )

        self.toolbox.register(
            "select",
            tools.selTournament,
            tournsize=CONFIG["tournament_size"]
        )

        # Register evaluation function
        if self.fitness_function:
            self.toolbox.register("evaluate", self.fitness_function)

    def set_fitness_function(self, fitness_function):
        """Set the fitness function to be used in evolution."""
        self.fitness_function = fitness_function
        self.toolbox.register("evaluate", fitness_function)

    def create_population(self, size):
        """Create a new population of the specified size."""
        return self.toolbox.population(n=size)

    def evaluate_population(self, population):
        """Evaluate the fitness of each individual in the population."""
        fitnesses = list(map(self.toolbox.evaluate, population))

        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        return population

    def evolve(self, population, generations=None, crossover_prob=None, mutation_prob=None, elite_size=None):
        """
        Evolve the population for a number of generations.

        Args:
            population: Initial population
            generations: Number of generations to evolve
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            elite_size: Number of best individuals to keep unchanged

        Returns:
            Evolved population, statistics
        """
        generations = generations or CONFIG["generations"]
        crossover_prob = crossover_prob or CONFIG["crossover_prob"]
        mutation_prob = mutation_prob or CONFIG["mutation_rate"]
        elite_size = elite_size or CONFIG["elite_size"]

        # Set up statistics to track
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)

        # Evolve using eaSimple algorithm with elitism
        population, log = algorithms.eaSimple(
            population=population,
            toolbox=self.toolbox,
            cxpb=crossover_prob,
            mutpb=mutation_prob,
            ngen=generations,
            stats=stats,
            halloffame=tools.HallOfFame(elite_size),
            verbose=True
        )

        return population, log

    def genome_to_network(self, genome):
        """
        Convert a genome (flat list of weights) to a neural network.

        Args:
            genome: Flat list of weights

        Returns:
            Neural network with the specified weights
        """
        nn = NeuralNetwork()
        nn.set_weights_flat(np.array(genome))
        return nn

    def network_to_genome(self, network):
        """
        Convert a neural network to a genome (flat list of weights).

        Args:
            network: Neural network

        Returns:
            Flat list of weights as a genome
        """
        return list(network.get_weights_flat())


class PyGADEvolution:
    """
    Handles evolutionary algorithms using PyGAD.
    Used for GPU-accelerated evolution of social strategies.
    """

    def __init__(self, genome_size=None, fitness_function=None):
        """
        Initialize the PyGAD evolution system.

        Args:
            genome_size: Size of the genome (if None, use from config)
            fitness_function: Function to evaluate fitness
        """
        self.genome_size = genome_size or DEAP_CONFIG["individual_size"]
        self.fitness_function = fitness_function
        self.use_metal = CONFIG["use_metal"] and HAS_PYGAD_METAL
        self.current_generation = 0

        # Will be created when needed
        self.ga_instance = None

    def _create_ga_instance(self, initial_population=None):
        """Create a PyGAD GA instance with configuration."""
        num_generations = PYGAD_CONFIG["num_generations"]
        num_parents_mating = PYGAD_CONFIG["num_parents_mating"]

        # Set up configuration
        ga_config = {
            "num_generations": num_generations,
            "num_parents_mating": num_parents_mating,
            "fitness_func": self.fitness_function,
            "sol_per_pop": CONFIG["population_size"],
            "num_genes": self.genome_size,
            "init_range_low": PYGAD_CONFIG["random_mutation_min_val"],
            "init_range_high": PYGAD_CONFIG["random_mutation_max_val"],
            "parent_selection_type": "tournament",
            "K_tournament": CONFIG["tournament_size"],
            "keep_elitism": CONFIG["elite_size"],
            "crossover_type": "single_point",
            "mutation_type": "random",
            "mutation_percent_genes": PYGAD_CONFIG["mutation_percent_genes"],
            "random_mutation_min_val": PYGAD_CONFIG["random_mutation_min_val"],
            "random_mutation_max_val": PYGAD_CONFIG["random_mutation_max_val"],
        }

        # Use Metal acceleration if available
        if self.use_metal:
            ga_config["torch_device"] = "mps"

        # Set initial population if provided
        if initial_population is not None:
            ga_config["initial_population"] = initial_population

        # Add parallel processing if configured
        if PYGAD_CONFIG["parallel_processing"]:
            ga_config["parallel_processing"] = ["thread", PYGAD_CONFIG["parallel_processing"]]

        # Create GA instance
        self.ga_instance = pygad.GA(**ga_config)

        return self.ga_instance

    def set_fitness_function(self, fitness_function):
        """Set the fitness function to be used in evolution."""
        self.fitness_function = fitness_function

        # Update GA instance if it exists
        if self.ga_instance:
            self.ga_instance.fitness_func = fitness_function

    def create_population(self, size):
        """Create a new population of the specified size."""
        return np.random.uniform(
            PYGAD_CONFIG["random_mutation_min_val"],
            PYGAD_CONFIG["random_mutation_max_val"],
            (size, self.genome_size)
        )

    def evolve(self, initial_population=None, generations=None):
        """
        Evolve the population for a number of generations.

        Args:
            initial_population: Initial population (optional)
            generations: Number of generations to evolve

        Returns:
            Best solution, final population
        """
        generations = generations or PYGAD_CONFIG["num_generations"]

        # Create or update GA instance
        if self.ga_instance is None or initial_population is not None:
            self._create_ga_instance(initial_population)

        # Run evolution
        self.ga_instance.run()

        # Get results
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        population = self.ga_instance.population

        return solution, population

    def genome_to_network(self, genome):
        """
        Convert a genome to a neural network.

        Args:
            genome: Numpy array of weights

        Returns:
            Neural network with the specified weights
        """
        nn = NeuralNetwork()
        nn.set_weights_flat(genome)
        return nn


# Factory functions to create the appropriate evolution system
def create_evolution_system(system_type="deap", genome_size=None, fitness_function=None):
    """
    Create an evolution system of the specified type.

    Args:
        system_type: "deap" or "pygad"
        genome_size: Size of genomes
        fitness_function: Function to evaluate fitness

    Returns:
        Evolution system instance
    """
    if system_type.lower() == "pygad" and CONFIG["use_pygad"]:
        return PyGADEvolution(genome_size, fitness_function)
    else:
        return DEAPEvolution(genome_size, fitness_function)


# Social strategy evolution

class SocialStrategyEvolution:
    """
    Specialized evolution system for social strategies.
    Combines genetic algorithms with reinforcement learning.
    """

    def __init__(self, strategy_size=None):
        """
        Initialize social strategy evolution.

        Args:
            strategy_size: Size of strategy genome
        """
        self.strategy_size = strategy_size or 10  # Default size

        # Create underlying evolution system
        # Use PyGAD if possible for GPU acceleration
        if CONFIG["use_pygad"] and HAS_PYGAD_METAL:
            self.evolution_system = create_evolution_system(
                "pygad", self.strategy_size, self._evaluate_strategy
            )
        else:
            self.evolution_system = create_evolution_system(
                "deap", self.strategy_size, self._evaluate_strategy
            )

        # Strategy history and cache
        self.strategy_history = []
        self.strategy_fitness_cache = {}

        # Current population
        self.population = None

    def _evaluate_strategy(self, strategy):
        """
        Evaluate a social strategy by simulating interactions.
        This is a placeholder - actual evaluation would depend on simulation.

        Args:
            strategy: Strategy genome to evaluate

        Returns:
            Fitness value or tuple
        """
        # This should be replaced with actual simulation-based evaluation
        # For now, return a placeholder fitness
        strategy_key = tuple(float(x) for x in strategy)

        if strategy_key in self.strategy_fitness_cache:
            return self.strategy_fitness_cache[strategy_key]

        # Placeholder: evaluate based on balance between cooperation and competition
        cooperation = sum(strategy[:len(strategy) // 2]) / (len(strategy) // 2)
        competition = sum(strategy[len(strategy) // 2:]) / (len(strategy) - len(strategy) // 2)

        # Reward strategies that have a good balance
        balance = 1.0 - abs(cooperation - competition)

        # Reward some level of both rather than extremes
        diversity = min(cooperation, competition) * 2

        fitness = (balance * 0.6 + diversity * 0.4,)

        # Cache the result
        self.strategy_fitness_cache[strategy_key] = fitness

        return fitness

    def set_evaluation_function(self, eval_function):
        """
        Set custom evaluation function for strategies.

        Args:
            eval_function: Function that takes a strategy and returns fitness
        """
        self.evolution_system.set_fitness_function(eval_function)

    def initialize_population(self, size=None):
        """
        Initialize a population of strategies.

        Args:
            size: Population size (optional)

        Returns:
            Initial population
        """
        size = size or CONFIG["population_size"]
        self.population = self.evolution_system.create_population(size)
        return self.population

    def evolve_generation(self):
        """
        Evolve the population for one generation.

        Returns:
            Best strategy from this generation
        """
        if self.population is None:
            self.initialize_population()

        # For DEAP
        if isinstance(self.evolution_system, DEAPEvolution):
            # Evaluate current population
            self.evolution_system.evaluate_population(self.population)

            # Get best strategy before evolution
            best_idx = np.argmax([ind.fitness.values[0] for ind in self.population])
            best_strategy = self.population[best_idx]

            # Evolve for one generation
            self.population, _ = self.evolution_system.evolve(
                self.population, generations=1
            )

            return best_strategy

        # For PyGAD
        elif isinstance(self.evolution_system, PyGADEvolution):
            if self.evolution_system.ga_instance is None:
                self.evolution_system._create_ga_instance(self.population)

            # Run for one generation
            self.evolution_system.ga_instance.num_generations = 1
            self.evolution_system.ga_instance.run()

            # Get best solution
            solution, solution_fitness, _ = self.evolution_system.ga_instance.best_solution()
            self.population = self.evolution_system.ga_instance.population

            return solution

    def get_best_strategy(self):
        """
        Get the best strategy from the current population.

        Returns:
            Best strategy genome
        """
        if self.population is None:
            return None

        # For DEAP
        if isinstance(self.evolution_system, DEAPEvolution):
            # Find individual with highest fitness
            best_idx = np.argmax([ind.fitness.values[0] for ind in self.population])
            return self.population[best_idx]

        # For PyGAD
        elif isinstance(self.evolution_system, PyGADEvolution):
            solution, _, _ = self.evolution_system.ga_instance.best_solution()
            return solution

    def strategy_to_behavior_weights(self, strategy):
        """
        Convert a strategy genome to behavior weights.

        Args:
            strategy: Strategy genome

        Returns:
            Dictionary of behavior weights
        """
        # Example mapping from strategy genes to behavior weights
        # This would be customized based on the actual simulation

        # Normalize to 0-1 range
        norm_strategy = np.array(strategy)
        norm_strategy = (norm_strategy - PYGAD_CONFIG["random_mutation_min_val"]) / (
                PYGAD_CONFIG["random_mutation_max_val"] - PYGAD_CONFIG["random_mutation_min_val"]
        )

        # Map to behavior weights
        behaviors = {
            "cooperation": norm_strategy[0],
            "aggression": norm_strategy[1],
            "exploration": norm_strategy[2],
            "risk_taking": norm_strategy[3],
            "loyalty": norm_strategy[4],
            "knowledge_sharing": norm_strategy[5],
            "resource_hoarding": norm_strategy[6],
            "status_seeking": norm_strategy[7],
            "alliance_building": norm_strategy[8],
            "conformity": norm_strategy[9] if len(norm_strategy) > 9 else 0.5
        }

        return behaviors