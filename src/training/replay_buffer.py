"""
Replay Buffer: Stores real experience for world model training.

DreamerV3 stores episodes and samples sequences for training.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
import random


class EpisodeBuffer:
    """
    Stores a single episode of experience.
    """

    def __init__(self):
        self.observations = []  # List of {'image': tensor}
        self.actions = []       # List of action indices
        self.rewards = []       # List of rewards
        self.dones = []         # List of done flags

    def add(
        self,
        obs: Dict[str, torch.Tensor],
        action: int,
        reward: float,
        done: bool,
    ):
        """Add a transition to the episode."""
        # Store observation (detached, on CPU)
        self.observations.append({
            'image': obs['image'].detach().cpu()
        })
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.observations)

    def get_sequence(
        self,
        start: int,
        length: int,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sequence from the episode.

        Returns:
            obs_seq: {'image': (T, C, H, W)}
            action_seq: (T,)
            reward_seq: (T,)
            done_seq: (T,)
        """
        end = min(start + length, len(self))

        # Stack observations
        images = torch.stack([self.observations[i]['image'] for i in range(start, end)])
        obs_seq = {'image': images}

        # Stack other data
        action_seq = torch.tensor(self.actions[start:end], dtype=torch.long)
        reward_seq = torch.tensor(self.rewards[start:end], dtype=torch.float32)
        done_seq = torch.tensor(self.dones[start:end], dtype=torch.bool)

        return obs_seq, action_seq, reward_seq, done_seq


class ReplayBuffer:
    """
    Replay buffer that stores episodes and samples sequences.

    Key features:
    - Stores complete episodes
    - Samples random sequences for training
    - Handles multiple parallel environments
    """

    def __init__(
        self,
        capacity: int = 1000000,  # Max transitions
        sequence_length: int = 50,
        min_sequences: int = 1,  # Min sequences before sampling
    ):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.min_sequences = min_sequences

        self.episodes = deque()  # Completed episodes
        self.current_episodes = {}  # In-progress episodes (by env index)
        self.total_transitions = 0

    def add(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        reset_mask: torch.Tensor,
    ):
        """
        Add transitions from vectorized environment.

        Args:
            obs: {'image': (num_envs, C, H, W)}
            actions: (num_envs,)
            rewards: (num_envs,)
            dones: (num_envs,)
            reset_mask: (num_envs,) - True where episode reset
        """
        num_envs = actions.shape[0]

        for i in range(num_envs):
            # Initialize episode buffer if needed
            if i not in self.current_episodes:
                self.current_episodes[i] = EpisodeBuffer()

            # Get single env data
            env_obs = {'image': obs['image'][i:i+1].squeeze(0)}
            action = actions[i].item()
            reward = rewards[i].item()
            done = dones[i].item()

            # Add to current episode
            self.current_episodes[i].add(env_obs, action, reward, done)
            self.total_transitions += 1

            # If episode ended, save it
            if done:
                if len(self.current_episodes[i]) >= self.sequence_length:
                    self.episodes.append(self.current_episodes[i])
                self.current_episodes[i] = EpisodeBuffer()

        # Remove old episodes if over capacity
        while self.total_transitions > self.capacity and len(self.episodes) > 1:
            removed = self.episodes.popleft()
            self.total_transitions -= len(removed)

    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device('cpu'),
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of sequences.

        Returns:
            obs_seq: {'image': (B, T, C, H, W)}
            action_seq: (B, T)
            reward_seq: (B, T)
            continue_seq: (B, T) - 1.0 where episode continues, 0.0 at end
            reset_mask: (B, T) - True at episode start
        """
        if len(self.episodes) < self.min_sequences:
            raise ValueError(f"Not enough episodes: {len(self.episodes)} < {self.min_sequences}")

        sequences = []
        for _ in range(batch_size):
            # Random episode
            episode = random.choice(self.episodes)

            # Random start position
            max_start = max(0, len(episode) - self.sequence_length)
            start = random.randint(0, max_start)

            # Get sequence
            obs_seq, action_seq, reward_seq, done_seq = episode.get_sequence(
                start, self.sequence_length
            )
            sequences.append((obs_seq, action_seq, reward_seq, done_seq))

        # Collate
        batch_obs = {'image': torch.stack([s[0]['image'] for s in sequences]).to(device)}
        batch_actions = torch.stack([s[1] for s in sequences]).to(device)
        batch_rewards = torch.stack([s[2] for s in sequences]).to(device)
        batch_dones = torch.stack([s[3] for s in sequences]).to(device)

        # Continue = 1 - done (but shifted: continue[t] = not done[t])
        batch_continues = (~batch_dones).float()

        # Reset mask: True at start of sequence
        batch_reset = torch.zeros_like(batch_dones)
        batch_reset[:, 0] = True

        return batch_obs, batch_actions, batch_rewards, batch_continues, batch_reset

    def __len__(self) -> int:
        """Number of stored episodes."""
        return len(self.episodes)

    @property
    def num_transitions(self) -> int:
        """Total stored transitions."""
        return self.total_transitions

    def is_ready(self) -> bool:
        """Check if buffer has enough data to sample."""
        return len(self.episodes) >= self.min_sequences
