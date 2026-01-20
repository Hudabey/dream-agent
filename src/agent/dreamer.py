"""
DreamerAgent: The complete DreamerV3 agent.

Combines:
- World Model (encoder, RSSM, decoder, predictors)
- Actor (policy)
- Critic (value function)

Trains by:
1. Collecting real experience
2. Training world model on real data
3. Imagining trajectories
4. Training actor/critic on dreams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from ..world_model import WorldModel
from .actor import Actor
from .critic import Critic, compute_lambda_returns
from .imagine import imagine_trajectory_with_grads, ImaginedTrajectory


@dataclass
class DreamerConfig:
    """Configuration for DreamerV3 agent."""
    # Image
    image_channels: int = 3
    image_size: Tuple[int, int] = (64, 64)

    # Actions
    action_dim: int = 15  # Procgen

    # World model
    depth: int = 48
    embed_dim: int = 1024
    deter_dim: int = 4096
    stoch_dim: int = 32
    stoch_classes: int = 32
    hidden_dim: int = 1024
    reward_bins: int = 255

    # Training
    imagination_horizon: int = 15
    gamma: float = 0.997
    lambda_: float = 0.95
    entropy_scale: float = 3e-4
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    world_model_lr: float = 1e-4

    # Target network
    slow_target: bool = True
    slow_target_fraction: float = 0.02


class DreamerAgent(nn.Module):
    """
    Complete DreamerV3 Agent.

    This is the main class that:
    - Interacts with the environment
    - Trains the world model
    - Learns to act through imagination
    """

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config

        # World Model
        self.world_model = WorldModel(
            image_channels=config.image_channels,
            image_size=config.image_size,
            action_dim=config.action_dim,
            depth=config.depth,
            embed_dim=config.embed_dim,
            deter_dim=config.deter_dim,
            stoch_dim=config.stoch_dim,
            stoch_classes=config.stoch_classes,
            hidden_dim=config.hidden_dim,
            reward_bins=config.reward_bins,
        )

        # Actor
        self.actor = Actor(
            state_dim=self.world_model.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        )

        # Critic
        self.critic = Critic(
            state_dim=self.world_model.state_dim,
            hidden_dim=config.hidden_dim,
            slow_target=config.slow_target,
            slow_target_fraction=config.slow_target_fraction,
        )

        # Current state (for acting)
        self._state = None

    def reset(self, batch_size: int = 1, device: torch.device = None):
        """Reset agent state for new episode."""
        if device is None:
            device = next(self.parameters()).device
        self._state = self.world_model.initial_state(batch_size, device)

    @torch.no_grad()
    def act(
        self,
        obs: Dict[str, torch.Tensor],
        prev_action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Select action given observation.

        Args:
            obs: {'image': (B, C, H, W)}
            prev_action: Previous action (None for first step)
            deterministic: Use argmax instead of sampling

        Returns:
            action: (B,) action indices
        """
        B = obs['image'].size(0)
        device = obs['image'].device

        # Initialize state if needed
        if self._state is None:
            self.reset(B, device)

        # Default prev_action
        if prev_action is None:
            prev_action = torch.zeros(B, dtype=torch.long, device=device)

        # Update state with observation
        self._state, _, _ = self.world_model.observe(obs, prev_action, self._state)

        # Get state vector and pick action
        state_vec = self.world_model.get_state_vector(self._state)
        action = self.actor.get_action(state_vec, deterministic=deterministic)

        return action

    @torch.no_grad()
    def get_value(self, obs: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """
        Get value estimate for current state.

        This is what we visualize in the demo!
        """
        if self._state is None:
            raise ValueError("Must call act() first to establish state")

        state_vec = self.world_model.get_state_vector(self._state)
        return self.critic.predict(state_vec)

    @torch.no_grad()
    def imagine_future(
        self,
        horizon: int = 15,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Imagine future frames from current state.

        This is for visualization - shows what the agent "dreams".

        Returns:
            imagined_frames: (B, H, C, H, W) decoded dream frames
            predicted_rewards: (B, H) expected rewards
            predicted_values: (B, H) value estimates
        """
        if self._state is None:
            raise ValueError("Must call act() first to establish state")

        # Imagine trajectory
        traj = imagine_trajectory_with_grads(
            self.world_model,
            self.actor,
            self.critic,
            self._state,
            horizon,
        )

        # Decode states to images
        imagined_frames = self.world_model.decoder(traj.state_vectors)['image']

        return imagined_frames, traj.rewards, traj.values[:, :-1]

    def train_world_model(
        self,
        obs_seq: Dict[str, torch.Tensor],
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        continue_seq: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Train world model on a batch of real experience."""
        return self.world_model.compute_loss(
            obs_seq, action_seq, reward_seq, continue_seq, reset_mask
        )

    def train_actor_critic(
        self,
        initial_states: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Train actor and critic on imagined trajectories.

        This is where DreamerV3 does most of its learning!
        """
        config = self.config

        # Imagine trajectories
        traj = imagine_trajectory_with_grads(
            self.world_model,
            self.actor,
            self.critic,
            initial_states,
            config.imagination_horizon,
        )

        # Compute lambda-returns for critic training
        with torch.no_grad():
            returns = compute_lambda_returns(
                traj.rewards,
                traj.values,
                traj.continues,
                config.gamma,
                config.lambda_,
            )

        # --- Critic loss ---
        # Predict values for all states
        critic_values = self.critic.predict(traj.state_vectors)
        critic_loss = F.mse_loss(critic_values, returns.detach())

        # --- Actor loss ---
        # Advantage = returns - baseline
        with torch.no_grad():
            baseline = self.critic.predict(traj.state_vectors, use_target=True)
            advantages = returns - baseline

        # Policy gradient with entropy bonus
        actor_loss = -(traj.log_probs * advantages.detach()).mean()
        entropy = self.actor.entropy(traj.state_vectors).mean()
        actor_loss = actor_loss - config.entropy_scale * entropy

        # Update target network
        self.critic.update_target()

        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy': entropy,
            'returns': returns.mean(),
            'advantages': advantages.mean(),
        }

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict for saving."""
        return {
            'world_model': self.world_model.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load state dict."""
        self.world_model.load_state_dict(state_dict['world_model'])
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
