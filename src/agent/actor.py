"""
Actor: Policy network that maps states to action distributions.
Learns entirely in imagination (dreams).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal
from typing import Union


class Actor(nn.Module):
    """
    Policy network for discrete action spaces (Procgen).

    Takes latent state vectors and outputs action distributions.
    Trained purely on imagined trajectories from the world model.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 15,  # Procgen
        hidden_dim: int = 1024,
        num_layers: int = 4,
        min_std: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.min_std = min_std

        # Build MLP
        layers = []
        in_dim = state_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim

        # Output layer (logits for discrete actions)
        layers.append(nn.Linear(hidden_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> Categorical:
        """
        Args:
            state: (B, state_dim) or (B, T, state_dim)
        Returns:
            Categorical distribution over actions
        """
        logits = self.net(state)
        return Categorical(logits=logits)

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action from state."""
        dist = self.forward(state)
        if deterministic:
            return dist.probs.argmax(dim=-1)
        return dist.sample()

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability of action."""
        dist = self.forward(state)
        return dist.log_prob(action)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Get entropy of action distribution."""
        dist = self.forward(state)
        return dist.entropy()


class ActorContinuous(nn.Module):
    """
    Policy network for continuous action spaces.
    (Not used for Procgen, but included for completeness)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        min_std: float = 0.1,
        max_std: float = 1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std

        # Shared backbone
        layers = []
        in_dim = state_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Mean and std heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Normal:
        """
        Args:
            state: (B, state_dim) or (B, T, state_dim)
        Returns:
            Normal distribution over actions
        """
        features = self.backbone(state)
        mean = self.mean_head(features)

        # Bounded std
        std = self.std_head(features)
        std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(std)

        return Normal(mean, std)
