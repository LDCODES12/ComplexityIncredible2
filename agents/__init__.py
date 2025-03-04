"""
Agent package for intelligent entities, neural networks, and evolution.
"""

from agents.agent import Agent
from agents.brain import NeuralNetwork, batch_predict
from agents.evolution import (
    DEAPEvolution,
    PyGADEvolution,
    SocialStrategyEvolution,
    create_evolution_system
)

__all__ = [
    'Agent',
    'NeuralNetwork',
    'batch_predict',
    'DEAPEvolution',
    'PyGADEvolution',
    'SocialStrategyEvolution',
    'create_evolution_system'
]