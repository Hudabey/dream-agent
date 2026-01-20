"""
RSSM: Recurrent State Space Model
The core of DreamerV3's world model - learns dynamics in latent space.

State has two parts:
1. Deterministic (h): Captures long-term dependencies via GRU
2. Stochastic (z): Captures uncertainty, sampled from categorical distribution

Key operations:
- Prior: Predict z from h alone (for imagination/dreaming)
- Posterior: Infer z from h AND observation (for training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, OneHotCategoricalStraightThrough
from typing import Tuple, Optional, Dict


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm - DreamerV3's trick for handling varying scales."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class RSSM(nn.Module):
    """
    Recurrent State Space Model.

    The heart of DreamerV3 - learns to predict future states.
    Uses categorical latents (discrete, but differentiable via straight-through).
    """

    def __init__(
        self,
        embed_dim: int = 1024,      # From encoder
        deter_dim: int = 4096,      # Deterministic state (GRU hidden)
        stoch_dim: int = 32,        # Number of categorical variables
        stoch_classes: int = 32,    # Classes per categorical
        hidden_dim: int = 1024,     # MLP hidden size
        action_dim: int = 15,       # Procgen has 15 actions
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.action_dim = action_dim

        # Full state dimension (for actor/critic)
        self.state_dim = deter_dim + stoch_dim * stoch_classes

        # GRU for deterministic state
        # Input: prev_stoch (flattened) + action
        gru_input_dim = stoch_dim * stoch_classes + action_dim
        self.gru = nn.GRUCell(gru_input_dim, deter_dim)

        # Prior network: h -> z (for dreaming)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_classes),
        )

        # Posterior network: h + embed -> z (for training)
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_classes),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Get initial state (zeros)."""
        return {
            'deter': torch.zeros(batch_size, self.deter_dim, device=device),
            'stoch': torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device),
        }

    def get_state_vector(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten state dict to vector for actor/critic."""
        deter = state['deter']  # (B, deter_dim) or (B, T, deter_dim)
        stoch = state['stoch']  # (B, stoch_dim, classes) or (B, T, stoch_dim, classes)

        # Flatten stochastic part
        if stoch.dim() == 3:
            stoch_flat = stoch.reshape(stoch.size(0), -1)
        else:  # (B, T, stoch_dim, classes)
            stoch_flat = stoch.reshape(*stoch.shape[:-2], -1)

        return torch.cat([deter, stoch_flat], dim=-1)

    def _get_stoch_dist(self, logits: torch.Tensor) -> OneHotCategoricalStraightThrough:
        """Create categorical distribution from logits."""
        logits = logits.reshape(*logits.shape[:-1], self.stoch_dim, self.stoch_classes)
        return OneHotCategoricalStraightThrough(logits=logits)

    def prior(self, deter: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute prior p(z|h) - predict stochastic state from deterministic only.
        Used during imagination/dreaming.

        Returns:
            state: Dict with 'deter' and 'stoch'
            prior_logits: For KL computation
        """
        logits = self.prior_net(deter)
        dist = self._get_stoch_dist(logits)
        stoch = dist.rsample()  # Straight-through gradient

        state = {'deter': deter, 'stoch': stoch}
        return state, logits

    def posterior(
        self,
        deter: torch.Tensor,
        embed: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute posterior q(z|h,o) - infer stochastic state from h AND observation.
        Used during training (has access to real observation).

        Returns:
            state: Dict with 'deter' and 'stoch'
            posterior_logits: For KL computation
        """
        x = torch.cat([deter, embed], dim=-1)
        logits = self.posterior_net(x)
        dist = self._get_stoch_dist(logits)
        stoch = dist.rsample()

        state = {'deter': deter, 'stoch': stoch}
        return state, logits

    def forward(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        embed: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        One step of RSSM.

        Args:
            prev_state: Previous state dict
            action: Action taken (B,) integers or (B, action_dim) one-hot
            embed: Observation embedding (if available, for posterior)

        Returns:
            state: New state dict
            prior_logits: Prior distribution logits
            posterior_logits: Posterior logits (None if no embed)
        """
        # One-hot encode action if needed
        if action.dim() == 1:
            action = F.one_hot(action, self.action_dim).float()

        # Flatten previous stochastic state
        prev_stoch_flat = prev_state['stoch'].reshape(prev_state['stoch'].size(0), -1)

        # GRU step: h' = GRU(h, [z, a])
        gru_input = torch.cat([prev_stoch_flat, action], dim=-1)
        deter = self.gru(gru_input, prev_state['deter'])

        # Prior (always computed)
        state, prior_logits = self.prior(deter)

        # Posterior (only if we have observation)
        posterior_logits = None
        if embed is not None:
            state, posterior_logits = self.posterior(deter, embed)

        return state, prior_logits, posterior_logits

    def imagine(
        self,
        initial_state: Dict[str, torch.Tensor],
        policy: nn.Module,
        horizon: int = 15,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Dream/imagine trajectories using learned dynamics and policy.
        This is where DreamerV3 does most of its learning!

        Args:
            initial_state: Starting state
            policy: Actor that maps state -> action distribution
            horizon: Number of steps to imagine

        Returns:
            states: Dict with 'deter' (B, T, D) and 'stoch' (B, T, S, C)
            actions: (B, T) imagined actions
            action_log_probs: (B, T) log probs of actions
        """
        B = initial_state['deter'].size(0)
        device = initial_state['deter'].device

        deters = []
        stochs = []
        actions = []
        log_probs = []

        state = initial_state

        for t in range(horizon):
            # Get state vector for policy
            state_vec = self.get_state_vector(state)

            # Sample action from policy
            action_dist = policy(state_vec)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            # Store
            deters.append(state['deter'])
            stochs.append(state['stoch'])
            actions.append(action)
            log_probs.append(log_prob)

            # Imagine next state (prior only - no observation!)
            state, _, _ = self.forward(state, action, embed=None)

        # Stack along time dimension
        states = {
            'deter': torch.stack(deters, dim=1),   # (B, T, deter_dim)
            'stoch': torch.stack(stochs, dim=1),   # (B, T, stoch_dim, classes)
        }
        actions = torch.stack(actions, dim=1)      # (B, T)
        log_probs = torch.stack(log_probs, dim=1)  # (B, T)

        return states, actions, log_probs
