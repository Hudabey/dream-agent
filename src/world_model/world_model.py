"""
WorldModel: The complete world model combining all components.

This is the "brain" that learns to simulate the environment.
It can:
1. Encode observations to latent space
2. Predict next states (dynamics)
3. Predict rewards
4. Predict episode termination
5. Decode latent states back to images (for visualization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .encoder import MultiEncoder
from .decoder import MultiDecoder
from .rssm import RSSM
from .predictors import RewardPredictor, ContinuePredictor, twohot_encode


class WorldModel(nn.Module):
    """
    Complete DreamerV3 World Model.

    Learns a latent dynamics model of the environment that can be used
    to "imagine" future trajectories without actually interacting with
    the environment.
    """

    def __init__(
        self,
        image_channels: int = 3,
        image_size: Tuple[int, int] = (64, 64),
        action_dim: int = 15,           # Procgen
        depth: int = 48,                # Conv channel multiplier
        embed_dim: int = 1024,          # Encoder output
        deter_dim: int = 4096,          # GRU hidden size
        stoch_dim: int = 32,            # Number of categorical vars
        stoch_classes: int = 32,        # Classes per categorical
        hidden_dim: int = 1024,         # MLP hidden size
        reward_bins: int = 255,         # For two-hot encoding
        kl_balance: float = 0.8,        # KL loss balancing
        kl_free: float = 1.0,           # Free nats
    ):
        super().__init__()

        self.kl_balance = kl_balance
        self.kl_free = kl_free

        # State dimension (for actor/critic)
        self.state_dim = deter_dim + stoch_dim * stoch_classes
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes

        # Components
        self.encoder = MultiEncoder(
            image_channels=image_channels,
            depth=depth,
            embed_dim=embed_dim,
            input_size=image_size,
        )

        self.decoder = MultiDecoder(
            state_dim=self.state_dim,
            image_channels=image_channels,
            depth=depth,
            output_size=image_size,
        )

        self.rssm = RSSM(
            embed_dim=embed_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            stoch_classes=stoch_classes,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        )

        self.reward_predictor = RewardPredictor(
            state_dim=self.state_dim,
            hidden_dim=hidden_dim,
            num_bins=reward_bins,
        )

        self.continue_predictor = ContinuePredictor(
            state_dim=self.state_dim,
            hidden_dim=hidden_dim,
        )

    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Get initial RSSM state."""
        return self.rssm.initial_state(batch_size, device)

    def get_state_vector(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten state dict to vector."""
        return self.rssm.get_state_vector(state)

    def encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observation to embedding."""
        return self.encoder(obs)

    def decode(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decode state to observation reconstruction."""
        state_vec = self.get_state_vector(state)
        return self.decoder(state_vec)

    def observe(
        self,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Process one observation (training mode - has access to real obs).

        Args:
            obs: Observation dict with 'image'
            action: Previous action
            state: Previous state (None for initial)

        Returns:
            new_state, prior_logits, posterior_logits
        """
        if state is None:
            B = obs['image'].size(0)
            state = self.initial_state(B, obs['image'].device)

        embed = self.encode(obs)
        new_state, prior_logits, posterior_logits = self.rssm(state, action, embed)

        return new_state, prior_logits, posterior_logits

    def imagine_step(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine one step forward (no observation - pure dreaming).

        Args:
            state: Current state
            action: Action to take

        Returns:
            next_state
        """
        next_state, _, _ = self.rssm(state, action, embed=None)
        return next_state

    def observe_sequence(
        self,
        obs_seq: Dict[str, torch.Tensor],
        action_seq: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a sequence of observations.

        Args:
            obs_seq: {'image': (B, T, C, H, W)}
            action_seq: (B, T) actions
            reset_mask: (B, T) - True where episode resets

        Returns:
            Dictionary with states, priors, posteriors, predictions
        """
        B, T = action_seq.shape
        device = action_seq.device

        # Storage
        deters = []
        stochs = []
        prior_logits_list = []
        posterior_logits_list = []

        state = self.initial_state(B, device)

        for t in range(T):
            # Reset state where episodes reset
            if t > 0:
                reset = reset_mask[:, t].unsqueeze(-1)
                state = {
                    'deter': state['deter'] * (1 - reset.float()),
                    'stoch': state['stoch'] * (1 - reset.unsqueeze(-1).float()),
                }

            # Get observation at this timestep
            obs_t = {'image': obs_seq['image'][:, t]}

            # Get action (use dummy action for t=0)
            if t == 0:
                action_t = torch.zeros(B, dtype=torch.long, device=device)
            else:
                action_t = action_seq[:, t - 1]

            # Observe
            state, prior_logits, posterior_logits = self.observe(obs_t, action_t, state)

            deters.append(state['deter'])
            stochs.append(state['stoch'])
            prior_logits_list.append(prior_logits)
            posterior_logits_list.append(posterior_logits)

        # Stack results
        states = {
            'deter': torch.stack(deters, dim=1),
            'stoch': torch.stack(stochs, dim=1),
        }
        prior_logits = torch.stack(prior_logits_list, dim=1)
        posterior_logits = torch.stack(posterior_logits_list, dim=1)

        return {
            'states': states,
            'prior_logits': prior_logits,
            'posterior_logits': posterior_logits,
        }

    def compute_loss(
        self,
        obs_seq: Dict[str, torch.Tensor],
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        continue_seq: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute world model training loss.

        Returns dict with:
            - total_loss
            - recon_loss (image reconstruction)
            - kl_loss (dynamics)
            - reward_loss
            - continue_loss
        """
        # Process sequence
        outputs = self.observe_sequence(obs_seq, action_seq, reset_mask)
        states = outputs['states']
        prior_logits = outputs['prior_logits']
        posterior_logits = outputs['posterior_logits']

        # Get state vectors for predictions
        state_vecs = self.get_state_vector(states)  # (B, T, state_dim)

        # --- Reconstruction loss ---
        recon = self.decoder(state_vecs)
        recon_loss = F.mse_loss(recon['image'], obs_seq['image'])

        # --- KL loss (dynamics) ---
        # DreamerV3 uses KL balancing
        prior_logits = prior_logits.reshape(*prior_logits.shape[:-1], self.stoch_dim, self.stoch_classes)
        posterior_logits = posterior_logits.reshape(*posterior_logits.shape[:-1], self.stoch_dim, self.stoch_classes)

        prior_probs = F.softmax(prior_logits, dim=-1)
        posterior_probs = F.softmax(posterior_logits, dim=-1)

        # KL(posterior || prior)
        kl_value = (posterior_probs * (
            F.log_softmax(posterior_logits, dim=-1) - F.log_softmax(prior_logits, dim=-1)
        )).sum(dim=-1).sum(dim=-1)  # Sum over classes and categoricals

        # Free nats
        kl_loss = torch.clamp(kl_value - self.kl_free, min=0).mean()

        # --- Reward loss ---
        reward_loss = self.reward_predictor.loss(state_vecs[:, 1:], reward_seq[:, :-1])

        # --- Continue loss ---
        continue_loss = self.continue_predictor.loss(state_vecs[:, 1:], continue_seq[:, :-1])

        # Total loss
        total_loss = recon_loss + kl_loss + reward_loss + continue_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'reward_loss': reward_loss,
            'continue_loss': continue_loss,
        }
