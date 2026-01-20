"""
Reward and Continue Predictors.
These predict rewards and episode termination from latent states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm for handling varying scales."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def twohot_encode(x: torch.Tensor, num_bins: int = 255, low: float = -20.0, high: float = 20.0) -> torch.Tensor:
    """
    Two-hot encoding for continuous values (DreamerV3 style).
    Distributes value between two adjacent bins.
    """
    # Symlog transform
    x = symlog(x)

    # Create bin edges
    bins = torch.linspace(low, high, num_bins, device=x.device)

    # Clamp to valid range
    x = x.clamp(low, high)

    # Find bin indices
    below = (bins <= x.unsqueeze(-1)).sum(dim=-1) - 1
    below = below.clamp(0, num_bins - 2)
    above = below + 1

    # Compute weights
    equal_bins = (bins[above] - bins[below]).clamp(min=1e-8)
    weight_above = (x - bins[below]) / equal_bins
    weight_below = 1.0 - weight_above

    # Create two-hot encoding
    target = torch.zeros(*x.shape, num_bins, device=x.device)
    target.scatter_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))

    return target


def twohot_decode(logits: torch.Tensor, num_bins: int = 255, low: float = -20.0, high: float = 20.0) -> torch.Tensor:
    """Decode two-hot distribution to scalar value."""
    bins = torch.linspace(low, high, num_bins, device=logits.device)
    probs = F.softmax(logits, dim=-1)
    value = (probs * bins).sum(dim=-1)
    return symexp(value)


class RewardPredictor(nn.Module):
    """
    Predicts rewards from latent states.
    Uses two-hot encoding (DreamerV3) for handling varying reward scales.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 1024,
        num_bins: int = 255,
    ):
        super().__init__()
        self.num_bins = num_bins

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim) or (B, T, state_dim)
        Returns:
            logits: (B, num_bins) or (B, T, num_bins)
        """
        return self.net(state)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Get scalar reward prediction."""
        logits = self.forward(state)
        return twohot_decode(logits, self.num_bins)

    def loss(self, state: torch.Tensor, target_reward: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss against two-hot encoded target."""
        logits = self.forward(state)
        target = twohot_encode(target_reward, self.num_bins)
        # Cross entropy between predicted and target distributions
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target * log_probs).sum(dim=-1)
        return loss.mean()


class ContinuePredictor(nn.Module):
    """
    Predicts episode continuation (1 - done) from latent states.
    Binary classification: will the episode continue?
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 1024,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim) or (B, T, state_dim)
        Returns:
            logits: (B, 1) or (B, T, 1)
        """
        return self.net(state)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Get continuation probability."""
        logits = self.forward(state)
        return torch.sigmoid(logits).squeeze(-1)

    def loss(self, state: torch.Tensor, target_continue: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy loss."""
        logits = self.forward(state).squeeze(-1)
        return F.binary_cross_entropy_with_logits(logits, target_continue.float())
