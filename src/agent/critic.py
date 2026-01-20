"""
Critic: Value function that estimates expected returns from states.

This is the VALUE FUNCTION that will be visualized in the demo.
It tells us "how good is this state?" - critical for interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..world_model.predictors import twohot_encode, twohot_decode, symlog, symexp


class Critic(nn.Module):
    """
    Value function network.

    Estimates V(s) = expected future returns from state s.
    Uses two-hot encoding like DreamerV3 for handling varying value scales.

    This is what we visualize as the "value heatmap" in the demo!
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_bins: int = 255,
        slow_target: bool = True,
        slow_target_fraction: float = 0.02,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.slow_target = slow_target
        self.slow_target_fraction = slow_target_fraction

        # Build MLP
        layers = []
        in_dim = state_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim

        # Output: bins for two-hot encoding
        layers.append(nn.Linear(hidden_dim, num_bins))

        self.net = nn.Sequential(*layers)

        # Target network (for stable training)
        if slow_target:
            self.target_net = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim) if i == 0 else
                nn.SiLU() if i % 2 == 1 else
                nn.Linear(hidden_dim, hidden_dim if i < (num_layers - 1) * 2 - 1 else num_bins)
                for i in range((num_layers - 1) * 2 + 1)
            ])
            # Copy parameters
            self._build_target()

    def _build_target(self):
        """Initialize target network as copy of main network."""
        self.target_net = type(self.net)(
            *[type(m)(m.in_features, m.out_features) if isinstance(m, nn.Linear) else type(m)()
              for m in self.net]
        )
        self.target_net.load_state_dict(self.net.state_dict())
        for param in self.target_net.parameters():
            param.requires_grad = False

    def update_target(self):
        """Soft update target network."""
        if not self.slow_target:
            return

        with torch.no_grad():
            for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                target_param.data.lerp_(param.data, self.slow_target_fraction)

    def forward(self, state: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim) or (B, T, state_dim)
            use_target: Use target network for stable estimates
        Returns:
            logits: (B, num_bins) or (B, T, num_bins)
        """
        net = self.target_net if (use_target and self.slow_target) else self.net
        return net(state)

    def predict(self, state: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """
        Get scalar value prediction.

        This is what we show in the value heatmap!

        Args:
            state: (B, state_dim) or (B, T, state_dim)
        Returns:
            value: (B,) or (B, T) scalar values
        """
        logits = self.forward(state, use_target)
        return twohot_decode(logits, self.num_bins)

    def loss(
        self,
        state: torch.Tensor,
        target_value: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss against target values.

        Args:
            state: (B, T, state_dim)
            target_value: (B, T) target values
            weight: Optional (B, T) importance weights
        """
        logits = self.forward(state)  # (B, T, num_bins)
        target = twohot_encode(target_value, self.num_bins)  # (B, T, num_bins)

        # Cross-entropy loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target * log_probs).sum(dim=-1)  # (B, T)

        if weight is not None:
            loss = loss * weight

        return loss.mean()


def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    gamma: float = 0.997,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """
    Compute lambda-returns for value learning.

    This combines TD learning with Monte Carlo returns for
    a good bias-variance tradeoff.

    Args:
        rewards: (B, T) rewards
        values: (B, T+1) value estimates (includes bootstrap)
        continues: (B, T) continuation probabilities
        gamma: Discount factor
        lambda_: Lambda for TD(lambda)

    Returns:
        returns: (B, T) lambda-returns
    """
    T = rewards.shape[1]
    returns = []

    # Bootstrap value
    last_return = values[:, -1]

    for t in reversed(range(T)):
        # TD target
        td_target = rewards[:, t] + gamma * continues[:, t] * values[:, t + 1]

        # Lambda-return
        last_return = td_target + gamma * lambda_ * continues[:, t] * (last_return - values[:, t + 1])
        returns.append(last_return)

    # Reverse to get correct order
    returns = torch.stack(returns[::-1], dim=1)
    return returns
