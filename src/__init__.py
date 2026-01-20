"""
Dream Agent - DreamerV3 on Procgen with Interpretable Visualization

This package implements DreamerV3 for Procgen environments,
with tools to visualize the agent's "dreams" and value function.
"""

from .agent import DreamerAgent, DreamerConfig
from .world_model import WorldModel
from .training import Trainer
from .envs import make_procgen_env

__version__ = "0.1.0"

__all__ = [
    'DreamerAgent',
    'DreamerConfig',
    'WorldModel',
    'Trainer',
    'make_procgen_env',
]
